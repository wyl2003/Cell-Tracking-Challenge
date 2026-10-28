"""Where do the remaining errors actually come from?

`src.diagnostics` says *how much* each AOGM term costs. This script says *what
those terms are made of*, which is what decides the next experiment:

  - Are the false negatives the DIM cells (so contrast normalisation helps),
    or cells touching the image border / entering the field (so it does not)?
  - Are the false positives fragments of an already-detected cell (so a merge
    step helps), or objects on real background (so a threshold helps)?

Matching rule follows CTC: a GT object is detected when some result object
overlaps it with IoU >= 0.5. Everything else is an FN; result objects that do
not match any GT object are FPs.

    python -m src.error_analysis --seq data/Fluo-N2DH-GOWT1/01 --res runs/01_RES_cpsam
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import tifffile

from . import ctc_io


def objects(mask: np.ndarray) -> dict[int, np.ndarray]:
    """label -> boolean pixel mask."""
    out = {}
    for label in np.unique(mask):
        if label:
            out[int(label)] = mask == label
    return out


def match_frame(gt: np.ndarray, res: np.ndarray, iou_threshold: float = 0.5):
    """Return (matched pairs, fn labels, fp labels) for one frame."""
    gt_objs, res_objs = objects(gt), objects(res)
    if not gt_objs or not res_objs:
        return [], sorted(gt_objs), sorted(res_objs)

    # Contingency counts via bincount on the combined label image.
    pairs, counts = np.unique(
        np.stack([gt.ravel(), res.ravel()], axis=1), axis=0, return_counts=True
    )
    inter = {}
    for (g, r), c in zip(pairs, counts):
        if g and r:
            inter[(int(g), int(r))] = int(c)
    gt_area = {g: int(m.sum()) for g, m in gt_objs.items()}
    res_area = {r: int(m.sum()) for r, m in res_objs.items()}

    matched, used_res = [], set()
    for (g, r), c in sorted(inter.items(), key=lambda kv: -kv[1]):
        if g in gt_area and r in res_area and r not in used_res:
            iou = c / (gt_area[g] + res_area[r] - c)
            if iou >= iou_threshold:
                matched.append((g, r, iou))
                used_res.add(r)
    matched_gt = {g for g, _, _ in matched}
    return matched, sorted(set(gt_objs) - matched_gt), sorted(set(res_objs) - used_res)


def border_distance(pixel_mask: np.ndarray) -> int:
    """Pixels from the nearest image edge to the object (0 = touches border)."""
    h, w = pixel_mask.shape
    ys, xs = np.nonzero(pixel_mask)
    return int(min(ys.min(), xs.min(), h - 1 - ys.max(), w - 1 - xs.max()))


def analyse(seq_dir: str | Path, res_dir: str | Path, iou_threshold: float = 0.5) -> dict:
    seq_dir, res_dir = Path(seq_dir), Path(res_dir)
    gt_dir = seq_dir.with_name(seq_dir.name + "_GT")  # data/.../01 -> data/.../01_GT
    gt_masks = ctc_io.list_masks(gt_dir / "TRA")
    res_masks = ctc_io.list_masks(res_dir)
    track_table = ctc_io.read_track_table(gt_dir / "TRA" / "man_track.txt")
    birth = {int(r[0]): int(r[1]) for r in track_table}

    frames = sorted(set(gt_masks) & set(res_masks))

    stats = {
        "frames": len(frames),
        "gt_total": 0, "res_total": 0, "tp": 0,
        "fn_intensity": [], "fn_area": [], "fn_border": [],
        "fn_on_birth": 0, "fn_total": 0,
        "fp_area": [], "fp_overlaps_gt": 0, "fp_border": [], "fp_total": 0,
        "gt_intensity_all": [],
    }
    area_ratio = []
    matched_gt_area: list[int] = []

    for frame in frames:
        gt = ctc_io.read_mask(gt_masks[frame])
        res = ctc_io.read_mask(res_masks[frame])
        img = tifffile.imread(seq_dir / f"t{frame:03d}.tif").astype(np.float64)

        matched, fn, fp = match_frame(gt, res, iou_threshold)
        stats["gt_total"] += len(objects(gt))
        stats["res_total"] += len(objects(res))
        stats["tp"] += len(matched)

        gt_objs, res_objs = objects(gt), objects(res)

        # ---- false negatives: how bright / how big / near the border / just born
        for label in fn:
            pix = gt_objs[label]
            stats["fn_intensity"].append(float(img[pix].mean()))
            stats["fn_area"].append(int(pix.sum()))
            stats["fn_border"].append(border_distance(pix))
            b = birth.get(label)
            stats["fn_on_birth"] += int(b is not None and 0 <= frame - b <= 1)
            stats["fn_total"] += 1

        for label in gt_objs:
            stats["gt_intensity_all"].append(float(img[gt_objs[label]].mean()))

        # ---- false positives: fragments of a detected cell, or objects on background
        gt_any = gt > 0
        for label in fp:
            pix = res_objs[label]
            stats["fp_area"].append(int(pix.sum()))
            stats["fp_border"].append(border_distance(pix))
            overlap = float((pix & gt_any).sum()) / max(int(pix.sum()), 1)
            stats["fp_overlaps_gt"] += int(overlap > 0.3)
            stats["fp_total"] += 1

        for g, r, _ in matched:
            area_ratio.append(res_objs[r].sum() / max(gt_objs[g].sum(), 1))
            matched_gt_area.append(int(gt_objs[g].sum()))

    return {**stats, "area_ratio_matched": area_ratio, "matched_gt_area": matched_gt_area}


def summarise(name: str, s: dict, gt_ref_intensity: float | None = None) -> None:
    print(f"\n=== {name} ===")
    print(f"frames {s['frames']}   GT objects {s['gt_total']}   predicted {s['res_total']}   "
          f"matched {s['tp']}")
    print(f"FN {s['fn_total']} ({100 * s['fn_total'] / max(s['gt_total'], 1):.1f}% of GT objects)"
          f"   FP {s['fp_total']}")

    if s["fn_total"]:
        fn_i = np.array(s["fn_intensity"])
        fn_a = np.array(s["fn_area"])
        fn_b = np.array(s["fn_border"])
        all_i = np.array(s["gt_intensity_all"])
        print(f"\nFN object brightness (mean px intensity in the GT mask)")
        print(f"  detected GT objects: median {np.median(all_i):.1f}   "
              f"10th pct {np.percentile(all_i, 10):.1f}")
        print(f"  missed   GT objects: median {np.median(fn_i):.1f}   "
              f"10th pct {np.percentile(fn_i, 10):.1f}")
        dim_thr = np.percentile(all_i, 25)
        print(f"  {100 * (fn_i < dim_thr).mean():.0f}% of FN are dimmer than the "
              f"25th-percentile detected cell ({dim_thr:.0f})")
        matched_a = np.array(s["matched_gt_area"]) if s["matched_gt_area"] else np.array([0])
        print(f"  FN area    median {np.median(fn_a):.0f} px   "
              f"(GT objects that WERE detected: median {np.median(matched_a):.0f} px)")
        print(f"  FN touching border (<=8 px): {100 * (fn_b <= 8).mean():.0f}%")
        print(f"  FN born same/previous frame: {100 * s['fn_on_birth'] / s['fn_total']:.0f}%")

    if s["fp_total"]:
        fp_a = np.array(s["fp_area"])
        print(f"\nFP objects")
        print(f"  median area {np.median(fp_a):.0f} px   "
              f"25th pct {np.percentile(fp_a, 25):.0f}   max {fp_a.max()}")
        print(f"  overlapping a GT object >30%: "
              f"{100 * s['fp_overlaps_gt'] / s['fp_total']:.0f}%  "
              f"(these are fragments, not detections on empty background)")
        print(f"  touching border (<=8 px): "
              f"{100 * (np.array(s['fp_border']) <= 8).mean():.0f}%")

    ar = np.array(s["area_ratio_matched"])
    if ar.size:
        print(f"\nmatched-object area ratio res/GT: median {np.median(ar):.2f}x   "
              f"p10 {np.percentile(ar, 10):.2f}   p90 {np.percentile(ar, 90):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", required=True, help="sequence folder, e.g. data/Fluo-N2DH-GOWT1/01")
    ap.add_argument("--res", required=True, help="result folder, e.g. runs/01_RES_cpsam")
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()
    summarise(f"{args.res} vs {args.seq}", analyse(args.seq, args.res, args.iou))


if __name__ == "__main__":
    main()

"""Full AOGM error decomposition of a CTC result folder.

`src.evaluate` reports the three headline numbers (DET / SEG / TRA). This script
reports the *six error terms those numbers are made of*:

    AOGM = 5·NS + 10·FN + 1·FP + 1·ED + 1.5·EA + 1·EC

    NS  split/merge vertices   (one detection where two cells should be)
    FN  missing vertices       (a cell that was not detected at all)
    FP  spurious vertices      (a detection with no matching cell)
    ED  redundant edge         (a link that should not exist)
    EA  missing edge           (a real link that was not made)
    EC  wrong edge semantics   (link exists but points at the wrong parent)

Two reasons this exists:

1. Score differences are not interpretable without it. "TRA went up by 0.03"
   means nothing; "FN fell by 41 and EA rose by 12" says exactly what changed.
2. SEG is recall-dominated, so it hides over-segmentation. The per-object area
   comparison printed here is the direct evidence (CTC reference nuclei on this
   dataset are ~1790 px; a segmenter that outputs 2700 px is not "almost right").

Usage
-----
    python -m src.diagnostics --gt data/Fluo-N2DH-GOWT1/01_GT --res runs/01_RES

Add `--json path.json` to also dump the raw dictionary (useful for building the
results table without re-running evaluation).
"""

from __future__ import annotations

import argparse
import json
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from . import ctc_io
from .evaluate import _prepare

# Weights read from ctc_metrics/metrics/technical/tra.py (py-ctcmetrics 1.3.3).
WEIGHTS = {"NS": 5.0, "FN": 10.0, "FP": 1.0, "ED": 1.0, "EA": 1.5, "EC": 1.0}

# Metrics worth pulling: the three headline numbers, the lineage metrics, and
# the division recall (BC(0) with i=0 = exact frame accuracy).
METRICS = ["DET", "SEG", "TRA", "CT", "TF", "BC(0)"]


def decompose(gt_dir: str | Path, res_dir: str | Path) -> dict:
    """Run the official metrics and return everything including the AOGM terms."""
    from ctc_metrics.scripts.evaluate import evaluate_sequence

    with tempfile.TemporaryDirectory() as tmp:
        gt_tmp, res_tmp = _prepare(Path(gt_dir), Path(res_dir), Path(tmp))
        buf = StringIO()
        with redirect_stdout(buf):  # evaluate_sequence is chatty
            # threads=1 on purpose: the multiprocessing path (threads=0) is
            # Windows-hostile and was observed to hang instead of finishing.
            results = evaluate_sequence(str(res_tmp), str(gt_tmp), list(METRICS), threads=1)
    if results.get("Valid") == 0:
        raise RuntimeError(f"results are not valid per CTC rules:\n{buf.getvalue()}")
    return results


def mean_area(directory: str | Path) -> tuple[float, int]:
    """Median object area (px) and object count over every mask in a folder."""
    areas = []
    for path in ctc_io.list_masks(directory).values():
        mask = ctc_io.read_mask(path)
        counts = np.bincount(np.asarray(mask).ravel())
        areas.extend(counts[1:].tolist())
    areas = [a for a in areas if a > 0]
    return (float(np.median(areas)) if areas else 0.0, len(areas))


def report(gt_dir: str | Path, res_dir: str | Path) -> dict:
    gt_dir, res_dir = Path(gt_dir), Path(res_dir)
    r = decompose(gt_dir, res_dir)

    print(f"GT {gt_dir}\nRES {res_dir}\n")
    print(f"{'DET':>6} {r.get('DET', float('nan')):.4f}      "
          f"{'SEG':>6} {r.get('SEG', float('nan')):.4f}      "
          f"{'TRA':>6} {r.get('TRA', float('nan')):.4f}")
    if r.get("OP_CTB") is not None:
        print(f"OP_CTB {r['OP_CTB']:.4f}   (mean of SEG and TRA)")
    print()

    aogm, aogm0 = r.get("AOGM"), r.get("AOGM_0")
    print(f"AOGM = {aogm:.1f}   AOGM_0 = {aogm0:.1f}   (TRA = 1 - AOGM/AOGM_0)\n")
    print(f"{'term':>5} {'count':>7} {'weight':>7} {'cost':>9} {'share':>7}  meaning")
    meanings = {
        "NS": "two cells merged into one detection",
        "FN": "cell never detected",
        "FP": "spurious detection",
        "ED": "redundant link",
        "EA": "missing link (track breaks)",
        "EC": "wrong edge semantics (bad parent)",
    }
    for key in ("NS", "FN", "FP", "ED", "EA", "EC"):
        count = r.get(f"AOGM_{key}", 0) or 0
        cost = WEIGHTS[key] * count
        share = 100.0 * cost / aogm if aogm else 0.0
        print(f"{key:>5} {count:>7.0f} {WEIGHTS[key]:>7.1f} {cost:>9.1f} {share:>6.1f}%  {meanings[key]}")
    print()

    for key in ("CT", "TF"):
        if r.get(key) is not None:
            print(f"{key:>5} = {r[key]:.4f}   "
                  f"({'correct lineage' if key == 'CT' else 'correct track fraction'})")
    if r.get("gt_divisions") is not None:
        tp = r.get("tp_div(0)", 0) or 0
        fp = r.get("fp_div(0)", 0) or 0
        fn = r.get("fn_div(0)", 0) or 0
        total = r["gt_divisions"]
        print(f"divisions: {total} in GT, {tp} correct, {fn} missed, {fp} spurious"
              f"  -> recall {tp / total:.2f}" if total else "")
    print()

    gt_area, gt_n = mean_area(gt_dir / "SEG") if (gt_dir / "SEG").is_dir() else (0.0, 0)
    res_area, res_n = mean_area(res_dir)
    if gt_area:
        print(f"object area (median px): GT {gt_area:.0f} (n={gt_n})   "
              f"RES {res_area:.0f} (n={res_n})   ratio {res_area / gt_area:.2f}x")
        print("  SEG is recall-dominated: a mask 1.5x too large still counts as detected,")
        print("  so this ratio is where over-segmentation actually shows up.")

    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", required=True, help="<seq>_GT directory")
    ap.add_argument("--res", required=True, help="<seq>_RES directory")
    ap.add_argument("--json", default=None, help="also write the raw results dict here")
    args = ap.parse_args()

    out = report(args.gt, args.res)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        # numpy scalars (int64/float64) come back from the metric library
        safe = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in out.items()}
        Path(args.json).write_text(json.dumps(safe, indent=2), encoding="utf-8")
        print(f"\nraw results -> {args.json}")


if __name__ == "__main__":
    main()

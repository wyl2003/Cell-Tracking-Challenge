"""Turn per-frame instance masks into a CTC result folder (mask###.tif + res_track.txt).

Example
-------
    python -m src.run_track --masks runs/cellpose/01 --out runs/01_RES --iou 0.2

`--masks` must contain one 16-bit label image per frame (any file name that ends
with the frame number, e.g. mask000.tif, t001.tif, or Cellpose output renamed).
Use `--from-gt` to track the gold SEG masks instead, which isolates tracking
quality from segmentation quality.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import ctc_io
from .track import frame_stats, track_sequence


def load_from_gt(seq_dir: str | Path) -> dict[int, np.ndarray]:
    """Read <seq>_GT/SEG/man_seg###.tif as the segmentation input."""
    return {f: ctc_io.read_mask(p) for f, p in ctc_io.list_masks(Path(seq_dir) / "SEG").items()}


def save_result(out: str | Path, relabelled: dict[int, np.ndarray], table: np.ndarray) -> None:
    """Write masks + res_track.txt, renumbered from 0 as CTC expects."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    first = min(relabelled)
    for frame in sorted(relabelled):
        ctc_io.write_mask(out / f"mask{frame - first:03d}.tif", relabelled[frame])
    table = np.asarray(table).copy()
    table[:, 1:3] -= first
    ctc_io.write_track_table(out / "res_track.txt", table)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--masks", help="directory of per-frame instance masks")
    src.add_argument("--from-gt", help="path to a <seq>_GT directory (uses SEG masks)")
    ap.add_argument("--out", required=True, help="output directory, usually <seq>_RES")
    ap.add_argument("--iou", type=float, default=0.2, help="IoU threshold for linking")
    ap.add_argument("--max-gap", type=int, default=0, help="frames a track may disappear")
    ap.add_argument("--no-divisions", action="store_true", help="disable mitosis detection")
    ap.add_argument("--division-containment", type=float, default=0.5,
                    help="fraction of a daughter that must lie inside the mother mask")
    args = ap.parse_args()

    if args.from_gt:
        masks = load_from_gt(args.from_gt)
    else:
        masks = {f: ctc_io.read_mask(p) for f, p in ctc_io.list_masks(args.masks).items()}

    relabelled, table = track_sequence(
        masks,
        iou_threshold=args.iou,
        max_gap=args.max_gap,
        detect_divisions=not args.no_divisions,
        division_containment=args.division_containment,
    )

    out = Path(args.out)
    save_result(out, relabelled, table)
    first = min(relabelled)
    shifted = table.copy()
    shifted[:, 1:3] -= first
    problems = ctc_io.check_consistency({f - first: relabelled[f] for f in relabelled}, shifted)
    n_frames = len(relabelled)
    n_tracks = len(table)
    n_div = len({int(p) for p in table[:, 3] if p > 0})  # one division per mother
    areas = [s["area"] for m in relabelled.values() for s in frame_stats(m).values()]
    print(f"frames={n_frames} tracks={n_tracks} divisions={n_div} "
          f"median_area={np.median(areas):.0f}px -> {out}")
    if problems:
        print("WARNING:", *problems[:10], sep="\n  ")
    else:
        print("consistency: ok")


if __name__ == "__main__":
    main()

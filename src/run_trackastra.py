"""Link existing per-frame masks with Trackastra (learned association, no training).

This is the "SOTA linker" reference row of the ablation table. It takes exactly
the same input as `run_track.py` -- one instance mask per frame -- and writes the
same CTC output, so the two can be compared frame-for-frame:

    python -m src.run_trackastra --img data/Fluo-N2DH-GOWT1/01 --masks runs/cpsam/01 \\
        --out runs/01_RES_trackastra --model general_2d --mode greedy

`--model`:
    general_2d  trained on diverse live-cell data (the fair, non-leaking choice)
    ctc         trained on CTC training data -- treated as a leakage upper bound,
                report it separately and say so
`--mode`:
    greedy      greedy linking with division edges (fast, no extra deps)
    ilp         ILP linking via motile (slower, usually a bit better)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tifffile


def load_series(directory: str | Path) -> np.ndarray:
    """Read t000.tif ... (or mask000.tif ...) into a (T, Y, X) array, ordered by frame."""
    directory = Path(directory)
    paths = sorted(directory.glob("*.tif"), key=lambda p: int("".join(c for c in p.stem if c.isdigit())))
    if not paths:
        raise FileNotFoundError(f"no .tif files in {directory}")
    return np.stack([tifffile.imread(str(p)) for p in paths])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--img", required=True, help="raw image folder (t000.tif ...)")
    ap.add_argument("--masks", required=True, help="instance mask folder (mask000.tif ...)")
    ap.add_argument("--out", required=True, help="output <seq>_RES folder")
    ap.add_argument("--model", default="general_2d", choices=["general_2d", "ctc", "general_2d_w_SAM2_features"])
    ap.add_argument("--mode", default="greedy", choices=["greedy", "greedy_nodiv", "ilp"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from trackastra.model import Trackastra
    from trackastra.tracking import graph_to_ctc

    from . import ctc_io

    imgs = load_series(args.img)
    masks = load_series(args.masks).astype(np.int32)
    if imgs.shape[-2:] != masks.shape[-2:]:
        raise ValueError(f"image {imgs.shape} and mask {masks.shape} spatial dims differ")
    print(f"imgs {imgs.shape} masks {masks.shape} labels/frame "
          f"{np.mean([len(np.unique(m)) - 1 for m in masks]):.1f}")

    t0 = time.time()
    model = Trackastra.from_pretrained(args.model, device=args.device)
    print(f"model {args.model} loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    graph, masks_tracked = model.track(imgs, masks, mode=args.mode)
    print(f"linked {len(graph.nodes)} nodes / {len(graph.edges)} edges in {time.time() - t0:.1f}s")

    # graph_to_ctc() writes man_track.txt + man_track####.tif, which is its own
    # convention rather than a CTC result folder. Write mask###.tif +
    # res_track.txt through ctc_io so this output is interchangeable with
    # run_track.py's, and so src.evaluate can score it unchanged.
    tracks, ctc_masks = graph_to_ctc(graph, masks_tracked)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for frame, mask in enumerate(ctc_masks):
        ctc_io.write_mask(out / f"mask{frame:03d}.tif", np.asarray(mask))
    table = tracks[["label", "t1", "t2", "parent"]].to_numpy(dtype=np.int64)
    ctc_io.write_track_table(out / "res_track.txt", table)

    n_div = int((table[:, 3] > 0).sum() // 2)  # one division = two children
    print(f"tracks={len(table)} divisions={n_div} -> {out}")


if __name__ == "__main__":
    main()

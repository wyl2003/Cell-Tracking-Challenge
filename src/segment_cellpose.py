"""Zero-shot Cellpose segmentation -> CTC mask###.tif (the group's baseline segmentation).

Cellpose 3.x is pinned on purpose: the 4.x default model (`cpsam`) ships 1.2 GB of
weights and takes minutes per 1024x1024 frame on CPU, which makes the full
ablation matrix impossible on a laptop. `nuclei` is 25 MB and runs at ~2.5 s/frame.

Example
-------
    python -m src.segment_cellpose --seq data/Fluo-N2DH-GOWT1/01 --out runs/cellpose/01

Speed at 1024x1024, diameter=48: 0.46 s/frame on an RTX 3060 Laptop, 2.0-2.6 s/frame
on a 20-core CPU. GPU is used automatically when torch reports CUDA available.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import tifffile


def segment_sequence(
    seq_dir: str | Path,
    out_dir: str | Path,
    model_type: str = "nuclei",
    diameter: float | None = 48.0,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    min_size: int = 200,
    use_gpu: bool | None = None,
) -> list[int]:
    import torch
    from cellpose import models

    if use_gpu is None:
        use_gpu = torch.cuda.is_available()
    print(f"device: {'cuda' if use_gpu else 'cpu'}", flush=True)

    seq_dir, out_dir = Path(seq_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = models.CellposeModel(gpu=use_gpu, model_type=model_type)
    frames = sorted(glob.glob(str(seq_dir / "t*.tif")))
    if not frames:
        raise FileNotFoundError(f"no t*.tif frames in {seq_dir}")

    counts = []
    for i, path in enumerate(frames):
        img = tifffile.imread(path)
        masks, _, _ = model.eval(
            img,
            diameter=diameter,
            channels=[0, 0],
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
        )
        masks = masks.astype(np.int32)
        tifffile.imwrite(str(out_dir / f"mask{i:03d}.tif"), masks.astype(np.uint16), compression="zlib")
        counts.append(int(masks.max()))
        print(f"{out_dir.name} {i:03d}: {masks.max()} objects", flush=True)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="nuclei")
    ap.add_argument("--diameter", type=float, default=48.0, help="0 lets Cellpose estimate it")
    ap.add_argument("--flow-threshold", type=float, default=0.4)
    ap.add_argument("--cellprob-threshold", type=float, default=0.0)
    ap.add_argument("--min-size", type=int, default=200)
    ap.add_argument("--cpu", action="store_true", help="force CPU even when CUDA is available")
    args = ap.parse_args()

    counts = segment_sequence(
        args.seq, args.out, args.model, args.diameter or None,
        args.flow_threshold, args.cellprob_threshold, args.min_size,
        use_gpu=False if args.cpu else None,
    )
    print(f"{len(counts)} frames -> {args.out}")
    print(f"instances/frame: min={min(counts)} median={int(np.median(counts))} max={max(counts)}")


if __name__ == "__main__":
    main()

"""Classical (trainable-parameter-free) segmentation baseline for CTC 2D fluorescent data.

Pipeline per frame, all steps ablatable from the command line:

    Gaussian smooth -> Otsu threshold -> hole fill -> remove small debris
    -> distance transform -> local maxima seeds -> watershed split
    -> drop small/faint instances -> relabel 1..N

This exists for two reasons:

1. It is a *reference floor*. Any deep model that cannot beat this is not worth
   the write-up, and the rubric wants a baseline that is clearly not your own work.
2. Its failures are the ones worth fixing. Thresholding collapses touching
   nuclei and drops dim ones; a fine-tuned model attacks exactly those.

Output is a directory of ``mask###.tif`` (one 16-bit label image per frame) that
``src.run_track`` consumes directly.

Example
-------
    python -m src.segment_classical --seq data/Fluo-N2DH-GOWT1/01 --out runs/classical/01
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import tifffile
from scipy import ndimage as ndi


def _otsu_foreground(img: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth, normalise brightness, then split with Otsu."""
    smooth = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma)
    # Per-frame normalisation: absolute intensities differ a lot between the two
    # GOWT1 sequences (max 221 vs 155), so a global threshold would not transfer.
    hi = np.percentile(smooth, 99.5)
    norm = np.clip(smooth / max(hi, 1e-6), 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    _, binary = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = binary > 0
    return ndi.binary_fill_holes(binary)


def _seeds(dist: np.ndarray, min_distance: int, min_peak: float) -> np.ndarray:
    """Local maxima of the distance transform, one marker per nucleus."""
    size = 2 * min_distance + 1
    peak = dist == ndi.maximum_filter(dist, size=size)
    peak &= dist >= min_peak
    markers, n = ndi.label(peak)
    return markers, n


def _watershed(binary: np.ndarray, markers: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Split the foreground along distance-transform valleys."""
    # cv2.watershed treats the image as a landscape and floods from markers, so we
    # hand it the inverted distance map: valleys between nuclei become ridges.
    landscape = np.clip(1.0 - dist / max(dist.max(), 1e-6), 0, 1)
    vis = np.repeat((landscape * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    ws = cv2.watershed(vis, markers.astype(np.int32))
    ws[~binary] = 0
    ws[ws < 0] = 0
    return ws


def segment_frame(
    img: np.ndarray,
    sigma: float = 1.5,
    min_area: int = 300,
    min_distance: int = 12,
    min_peak_frac: float = 0.22,
    split_touching: bool = True,
) -> np.ndarray:
    binary = _otsu_foreground(img, sigma)
    binary = ndi.binary_opening(binary, np.ones((3, 3)))
    if min_area > 0:
        lab, n = ndi.label(binary)
        sizes = ndi.sum(binary, lab, range(1, n + 1))
        keep = np.isin(lab, np.flatnonzero(sizes >= min_area) + 1)
        binary = keep

    dist = ndi.distance_transform_edt(binary)
    markers, n = _seeds(dist, min_distance, min_peak_frac * dist.max() if dist.max() else 0)

    if split_touching and n > 0:
        out = _watershed(binary, markers, dist)
    else:
        out = ndi.label(binary)[0]

    # final pass: drop instances that ended up tiny after splitting
    if min_area > 0:
        lab, k = ndi.label(out > 0)
        sizes = ndi.sum(out > 0, lab, range(1, k + 1))
        keep = np.isin(lab, np.flatnonzero(sizes >= min_area) + 1)
        out = np.where(keep, out, 0)

    uniq = np.unique(out)
    uniq = uniq[uniq > 0]
    remap = np.zeros(int(out.max()) + 1, dtype=np.int32)
    remap[uniq] = np.arange(1, len(uniq) + 1)
    return remap[out]


def segment_sequence(seq_dir: str | Path, out_dir: str | Path, **kwargs) -> list[int]:
    """Segment every t###.tif in seq_dir; returns instance counts per frame."""
    seq_dir, out_dir = Path(seq_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(glob.glob(str(seq_dir / "t*.tif")))
    if not frames:
        raise FileNotFoundError(f"no t*.tif frames in {seq_dir}")
    counts = []
    for i, path in enumerate(frames):
        mask = segment_frame(tifffile.imread(path), **kwargs)
        tifffile.imwrite(str(out_dir / f"mask{i:03d}.tif"), mask.astype(np.uint16), compression="zlib")
        counts.append(int(mask.max()))
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", required=True, help="sequence directory with t###.tif frames")
    ap.add_argument("--out", required=True, help="output directory for mask###.tif")
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--min-area", type=int, default=300)
    ap.add_argument("--min-distance", type=int, default=12, help="seed spacing; ~half a nucleus radius")
    ap.add_argument("--min-peak-frac", type=float, default=0.22, help="seed minimum as a fraction of max distance")
    ap.add_argument("--no-split", action="store_true", help="skip watershed (ablation: touching cells stay merged)")
    args = ap.parse_args()

    counts = segment_sequence(
        args.seq,
        args.out,
        sigma=args.sigma,
        min_area=args.min_area,
        min_distance=args.min_distance,
        min_peak_frac=args.min_peak_frac,
        split_touching=not args.no_split,
    )
    print(f"{len(counts)} frames -> {args.out}")
    print(f"instances/frame: min={min(counts)} median={int(np.median(counts))} max={max(counts)}")


if __name__ == "__main__":
    main()

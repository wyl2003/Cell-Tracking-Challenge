"""Evaluate a CTC result folder against the gold annotations with py-ctcmetrics.

    pip install py-ctcmetrics
    python -m src.evaluate --gt data/Fluo-N2DL-HeLa/01_GT --res runs/01_RES

Both directories are renumbered to 0-based mask indices in a temporary folder
first: the metric code looks up `comp_masks[int(suffix)]`, so a sequence whose
frames are numbered from 1 would otherwise be scored one frame off (or fail).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from . import ctc_io


def _prepare(gt_dir: Path, res_dir: Path, work: Path) -> tuple[Path, Path]:
    gt_out, res_out = work / "GT", work / "RES"
    (gt_out / "TRA").mkdir(parents=True, exist_ok=True)
    (gt_out / "SEG").mkdir(parents=True, exist_ok=True)
    res_out.mkdir(parents=True, exist_ok=True)

    tra_masks = ctc_io.list_masks(gt_dir / "TRA")
    mapping = {old: new for new, old in enumerate(tra_masks)}
    for old, new in mapping.items():
        ctc_io.write_mask(gt_out / "TRA" / f"man_track{new:03d}.tif", ctc_io.read_mask(tra_masks[old]))
    seg_dir = gt_dir / "SEG"
    if seg_dir.is_dir():
        for old, new in mapping.items():
            src = seg_dir / f"man_seg{old:03d}.tif"
            if not src.exists():  # tolerate other suffixes
                hits = [p for p in seg_dir.glob("*.tif") if ctc_io.frame_index(p) == old]
                if not hits:
                    continue
                src = hits[0]
            ctc_io.write_mask(gt_out / "SEG" / f"man_seg{new:03d}.tif", ctc_io.read_mask(src))
    shutil.copy(gt_dir / "TRA" / "man_track.txt", gt_out / "TRA" / "man_track.txt")

    res_masks = ctc_io.list_masks(res_dir)
    first = min(res_masks)
    for frame, path in res_masks.items():
        ctc_io.write_mask(res_out / f"mask{frame - first:03d}.tif", ctc_io.read_mask(path))
    table = ctc_io.read_track_table(res_dir / "res_track.txt")
    table[:, 1:3] -= first
    ctc_io.write_track_table(res_out / "res_track.txt", table)
    return gt_out, res_out


def evaluate(gt_dir: str | Path, res_dir: str | Path, metrics=("DET", "SEG", "TRA")) -> dict[str, float]:
    gt_dir, res_dir = Path(gt_dir), Path(res_dir)
    with tempfile.TemporaryDirectory() as tmp:
        gt_tmp, res_tmp = _prepare(gt_dir, res_dir, Path(tmp))
        cmd = [sys.executable, "-m", "ctc_metrics.scripts.evaluate",
               "--gt", str(gt_tmp), "--res", str(res_tmp), "-n", "1"]
        cmd += [f"--{m.lower()}" for m in metrics]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        out = proc.stdout + proc.stderr
        if proc.returncode != 0:
            raise RuntimeError(f"ctc_evaluate failed:\n{out}")
    values = {}
    for key in metrics:
        m = re.search(rf"{key}\s*[=:]\s*([0-9.]+)", out)
        if m:
            values[key] = float(m.group(1))
    if not values:
        print(out)
    return values


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", required=True, help="<seq>_GT directory")
    ap.add_argument("--res", required=True, help="<seq>_RES directory")
    ap.add_argument("--metrics", default="DET,SEG,TRA")
    args = ap.parse_args()
    values = evaluate(args.gt, args.res, tuple(m.strip().upper() for m in args.metrics.split(",")))
    for key, val in values.items():
        print(f"{key}: {val:.4f}")
    if {"SEG", "TRA"} <= values.keys():
        print(f"OP_CTB (mean of SEG and TRA): {(values['SEG'] + values['TRA']) / 2:.4f}")


if __name__ == "__main__":
    main()

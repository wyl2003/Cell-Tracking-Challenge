"""Evaluate a CTC result folder against the gold annotations with py-ctcmetrics.

    pip install py-ctcmetrics
    python -m src.baseline_tracking.evaluate --gt data/Fluo-N2DL-HeLa/01_GT --res runs/01_RES

Both directories are renumbered to 0-based mask indices in a temporary folder
first: the metric code looks up `comp_masks[int(suffix)]`, so a sequence whose
frames are numbered from 1 would otherwise be scored one frame off (or fail).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from . import ctc_io


def _normalise_table(table, masks, first):
    """Accept table times matching filenames or already zero-based, verified by masks."""
    zero_masks = {f - first: mask for f, mask in masks.items()}
    errors = []
    for offset in dict.fromkeys((first, 0)):
        candidate = table.copy()
        candidate[:, 1:3] -= offset
        problems = ctc_io.check_consistency(zero_masks, candidate)
        if not problems:
            return candidate
        errors.extend(problems)
    raise ValueError("cannot align table times with mask frames: " + "; ".join(errors[:5]))


def _prepare(gt_dir: Path, res_dir: Path, work: Path) -> tuple[Path, Path]:
    from .track import validate_masks

    gt_out, res_out = work / "GT", work / "RES"
    tra = {f: ctc_io.read_mask(p) for f, p in ctc_io.list_masks(gt_dir / "TRA").items()}
    res = {f: ctc_io.read_mask(p) for f, p in ctc_io.list_masks(res_dir).items()}
    validate_masks(tra)
    validate_masks(res)
    if len(tra) != len(res):
        raise ValueError(f"GT has {len(tra)} frames but RES has {len(res)}; do not score partial sequences")
    if next(iter(tra.values())).shape != next(iter(res.values())).shape:
        raise ValueError("GT and RES mask shapes differ")
    gt_first, res_first = min(tra), min(res)
    gt_table = _normalise_table(ctc_io.read_track_table(gt_dir / "TRA" / "man_track.txt"), tra, gt_first)
    res_table = _normalise_table(ctc_io.read_track_table(res_dir / "res_track.txt"), res, res_first)
    (gt_out / "TRA").mkdir(parents=True, exist_ok=True)
    (gt_out / "SEG").mkdir(parents=True, exist_ok=True)
    res_out.mkdir(parents=True, exist_ok=True)
    for f, mask in tra.items():
        ctc_io.write_mask(gt_out / "TRA" / f"man_track{f - gt_first:03d}.tif", mask)
    seg_dir = gt_dir / "SEG"
    if seg_dir.is_dir() and any(seg_dir.glob("*.tif")):
        for f, path in ctc_io.list_masks(seg_dir).items():
            if f not in tra:
                raise ValueError(f"SEG frame {f} is outside the TRA sequence")
            ctc_io.write_mask(gt_out / "SEG" / f"man_seg{f - gt_first:03d}.tif", ctc_io.read_mask(path))
    for f, mask in res.items():
        ctc_io.write_mask(res_out / f"mask{f - res_first:03d}.tif", mask)
    ctc_io.write_track_table(gt_out / "TRA" / "man_track.txt", gt_table)
    ctc_io.write_track_table(res_out / "res_track.txt", res_table)
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
    if len(values) != len(metrics):
        raise RuntimeError(f"requested metrics missing from evaluator output:\n{out}")
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

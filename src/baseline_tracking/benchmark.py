"""Controlled tracking ablations on ONE fixed segmentation sequence.

python -m src.baseline_tracking.benchmark --masks runs/cpsam/01 --gt data/Fluo-N2DH-GOWT1/01_GT --out runs/tracking_01
Parameters must be selected on development data before running a held-out sequence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np

from . import ctc_io
from .diagnostics import decompose
from .run_track import save_result
from .track import track_sequence, validate_masks
from ..track import track_sequence as legacy_tracker


def run_benchmark(mask_dir, gt_dir, out_dir, max_distance=30.0):
    paths = ctc_io.list_masks(mask_dir)
    masks = {f: ctc_io.read_mask(p) for f, p in paths.items()}
    validate_masks(masks)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    versions = {name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'tifffile', 'py-ctcmetrics']}
    manifest = {
        'mask_directory': str(Path(mask_dir).resolve()), 'gt_directory': str(Path(gt_dir).resolve()),
        'python': platform.python_version(), 'packages': versions,
        'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()},
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        'original_tracker_sha256': hashlib.sha256((Path(__file__).parent.parent / 'track.py').read_bytes()).hexdigest(),
        'note': 'One fixed mask input; no parameter selection or training is performed here.',
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    settings = [
        ('legacy_iou_div', legacy_tracker, {}),
        ('gated_iou_no_div', track_sequence, {'detect_divisions': False}),
        ('gated_iou_overlap_div', track_sequence, {}),
        ('hybrid_no_div', track_sequence, {'association': 'hybrid', 'detect_divisions': False, 'max_distance': max_distance}),
        ('iou_motion_div', track_sequence, {'division_mode': 'motion'}),
        ('hybrid_motion_div', track_sequence, {'association': 'hybrid', 'division_mode': 'motion', 'max_distance': max_distance}),
    ]
    rows = []
    def safe(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)

    for name, tracker, options in settings:
        config = dict(iou_threshold=0.2, max_gap=0, detect_divisions=True,
                      division_containment=0.5, area_ratio_range=(0.5, 1.8))
        config.update(options)
        if tracker is track_sequence:
            for key, value in dict(association='iou', max_distance=30.0,
                                   division_mode='overlap', division_distance=2.0).items():
                config.setdefault(key, value)
        start = time.perf_counter()
        tracked, table = tracker(masks, **config)
        elapsed = time.perf_counter() - start
        # Association-only experiments must preserve each input instance exactly.
        for f, original in masks.items():
            if not np.array_equal(original > 0, tracked[f] > 0):
                raise AssertionError(f'{name}: changed foreground in frame {f}')
            for label in np.unique(original):
                if label and len(np.unique(tracked[f][original == label])) != 1:
                    raise AssertionError(f'{name}: split an input instance')
            if len(np.unique(original)) != len(np.unique(tracked[f])):
                raise AssertionError(f'{name}: merged input instances')
        save_result(out / name, tracked, table)
        metrics = decompose(gt_dir, out / name)
        payload = {'method': name, 'parameters': config, 'tracking_seconds': elapsed, 'metrics': metrics}
        (out / f'{name}.json').write_text(json.dumps(payload, indent=2, default=safe), encoding='utf-8')
        tp, fp, fn = (metrics.get(k, 0) or 0 for k in ['tp_div(0)', 'fp_div(0)', 'fn_div(0)'])
        row = {'method': name, 'tracks': len(table), 'tracking_seconds': elapsed}
        for key in ['DET', 'SEG', 'TRA', 'CT', 'TF', 'BC(0)', 'AOGM', 'AOGM_NS', 'AOGM_FN', 'AOGM_FP', 'AOGM_ED', 'AOGM_EA', 'AOGM_EC']:
            row[key] = metrics.get(key)
        row.update(division_tp=tp, division_fp=fp, division_fn=fn,
                   division_precision=tp/(tp+fp) if tp+fp else None,
                   division_recall=tp/(tp+fn) if tp+fn else None)
        rows.append(row)
        with (out / 'summary.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{name}: TRA={metrics.get('TRA')} CT={metrics.get('CT')} TF={metrics.get('TF')}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--masks', required=True)
    parser.add_argument('--gt', required=True)
    parser.add_argument('--out', required=True, help='new directory; existing directories are refused')
    parser.add_argument('--max-distance', type=float, default=30)
    args = parser.parse_args()
    run_benchmark(args.masks, args.gt, args.out, args.max_distance)


if __name__ == '__main__':
    main()

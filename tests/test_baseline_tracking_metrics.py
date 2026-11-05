"""Synthetic stress cases checked with the official metrics, not just track IDs."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baseline_tracking import ctc_io
from src.baseline_tracking.diagnostics import decompose
from src.baseline_tracking.run_track import save_result
from src.baseline_tracking.track import track_sequence
from src.track import track_sequence as legacy_tracker
from test_baseline_tracking import frame


@pytest.mark.parametrize('case', ['fast_motion', 'crossing', 'displaced_division'])
def test_stress_metrics(tmp_path, case):
    if case == 'fast_motion':
        masks = {i: frame((1, 30, 10+10*i, 3)) for i in range(4)}
        table = np.array([[1, 0, 3, 0]])
        options = dict(association='hybrid', max_distance=12, detect_divisions=False)
    elif case == 'crossing':
        masks = {0: frame((1, 30, 10, 2), (2, 30, 40, 2)),
                 1: frame((1, 30, 20, 2), (2, 30, 30, 2)),
                 2: frame((1, 30, 30, 2), (2, 30, 20, 2))}
        table = np.array([[1, 0, 2, 0], [2, 0, 2, 0]])
        options = dict(association='hybrid', max_distance=12, detect_divisions=False)
    else:
        masks = {0: frame((1, 40, 40, 8)),
                 1: frame((2, 40, 28, 5), (3, 40, 52, 5))}
        table = np.array([[1, 0, 0, 0], [2, 1, 1, 1], [3, 1, 1, 1]])
        options = dict(division_mode='motion')
    gt = tmp_path / 'GT'
    for sub, prefix in [('TRA', 'man_track'), ('SEG', 'man_seg')]:
        for f, mask in masks.items():
            ctc_io.write_mask(gt / sub / f'{prefix}{f:03d}.tif', mask)
    ctc_io.write_track_table(gt / 'TRA/man_track.txt', table)
    # Deliberately destroy temporal label correspondence to avoid GT-ID leakage.
    inputs = {f: np.where(mask, mask + 10*(f+1), 0).astype(np.int32) for f, mask in masks.items()}
    scores = []
    for name, tracker, config in [('legacy', legacy_tracker, {}), ('new', track_sequence, options)]:
        out, predicted = tracker(inputs, **config)
        save_result(tmp_path / name, out, predicted)
        metrics = decompose(gt, tmp_path / name)
        assert metrics['DET'] == pytest.approx(1)
        assert metrics['SEG'] == pytest.approx(1)
        scores.append(metrics['TRA'])
    assert scores[1] == pytest.approx(1)
    assert scores[1] > scores[0]

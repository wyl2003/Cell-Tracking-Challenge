"""Regression tests for association, biological events and CTC output invariants."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baseline_tracking import ctc_io
from src.baseline_tracking.track import track_sequence, gated_assignment, overlap_matrix
from src.baseline_tracking.run_track import save_result, load_from_gt
from src.baseline_tracking.evaluate import _prepare
from test_pipeline import write_gt, build_sequence


def frame(*objects):
    mask = np.zeros((80, 80), dtype=np.int32)
    yy, xx = np.indices(mask.shape)
    for label, y, x, radius in objects:
        mask[(yy - y)**2 + (xx - x)**2 <= radius**2] = label
    return mask


def test_gating_before_assignment():
    # Old Hungarian chooses diagonal (.9+.19), then drops .19; off-diagonal
    # contains TWO legitimate links and must be available before assignment.
    scores = np.array([[.9, .5], [.5, .19]])
    assert set(gated_assignment(scores, scores >= .2)) == {(0, 1), (1, 0)}


def test_zero_iou_never_links_in_iou_mode():
    _, table = track_sequence({0: frame((3, 10, 10, 2)), 1: frame((8, 60, 60, 2))}, iou_threshold=0)
    assert len(table) == 2


def test_hybrid_links_fast_motion_without_overlap():
    masks = {i: frame((10+i, 30, 10+10*i, 3)) for i in range(4)}
    _, old = track_sequence(masks, detect_divisions=False)
    out, new = track_sequence(masks, association='hybrid', max_distance=12, detect_divisions=False)
    assert len(old) == 4 and len(new) == 1
    assert not ctc_io.check_consistency(out, new)


def test_velocity_preserves_crossing_identities():
    masks = {
        0: frame((1, 30, 10, 2), (2, 30, 40, 2)),
        1: frame((9, 30, 20, 2), (7, 30, 30, 2)),
        2: frame((8, 30, 30, 2), (4, 30, 20, 2)),
    }
    out, _ = track_sequence(masks, association='hybrid', max_distance=12, detect_divisions=False)
    assert out[0][30, 10] == out[2][30, 30]
    assert out[0][30, 40] == out[2][30, 20]


def test_motion_division_without_iou_continuation():
    masks = {0: frame((1, 40, 40, 8)),
             1: frame((5, 40, 28, 5), (9, 40, 52, 5))}
    _, baseline = track_sequence(masks)
    out, improved = track_sequence(masks, division_mode='motion')
    assert np.count_nonzero(baseline[:, 3]) == 0
    assert np.count_nonzero(improved[:, 3]) == 2
    assert not ctc_io.check_consistency(out, improved)


def test_two_continuing_cells_are_not_a_division():
    masks = {i: frame((1+i*2, 30, 20+i, 5), (2+i*2, 30, 40-i, 5)) for i in range(3)}
    _, table = track_sequence(masks, division_mode='motion', association='hybrid')
    assert len(table) == 2 and not table[:, 3].any()


def test_blank_frame_terminates_track():
    masks = {0: frame((1, 30, 30, 5)), 1: frame(), 2: frame((9, 30, 30, 5))}
    out, table = track_sequence(masks)
    assert len(table) == 2 and not ctc_io.check_consistency(out, table)
    with pytest.raises(ValueError, match='mask in every'):
        track_sequence(masks, max_gap=1)


def test_sparse_frames_rejected():
    with pytest.raises(ValueError, match='consecutive'):
        track_sequence({0: frame(), 2: frame()})


def test_internal_gap_detected():
    masks = {0: frame((1, 30, 30, 4)), 1: frame(), 2: frame((1, 30, 30, 4))}
    assert any('missing masks inside' in p for p in ctc_io.check_consistency(masks, np.array([[1, 0, 2, 0]])))


def test_invalid_parent_timing_detected():
    masks = {0: frame((1, 10, 10, 3)), 1: frame((1, 10, 10, 3), (2, 30, 30, 3))}
    assert any('parent 1 must end' in p for p in ctc_io.check_consistency(masks, np.array([[1, 0, 1, 0], [2, 1, 1, 1]])))


def test_duplicates_and_overflow(tmp_path):
    ctc_io.write_mask(tmp_path / 'mask000.tif', frame())
    ctc_io.write_mask(tmp_path / 't000.tif', frame())
    with pytest.raises(ValueError, match='duplicate frame'):
        ctc_io.list_masks(tmp_path)
    with pytest.raises(ValueError, match='overflow'):
        ctc_io.write_mask(tmp_path / 'overflow.tif', np.array([[65536]]))


def test_save_rejects_stale_result(tmp_path):
    out, table = track_sequence({0: frame((1, 30, 30, 4))})
    save_result(tmp_path, out, table)
    with pytest.raises(FileExistsError):
        save_result(tmp_path, out, table)


def test_empty_foreground_valid(tmp_path):
    out, table = track_sequence({1: frame(), 2: frame()})
    assert table.shape == (0, 4)
    save_result(tmp_path, out, table)


def test_from_gt_uses_tra(tmp_path):
    ctc_io.write_mask(tmp_path / 'TRA/man_track000.tif', frame((4, 30, 30, 3)))
    ctc_io.write_mask(tmp_path / 'SEG/man_seg000.tif', frame((7, 30, 30, 7)))
    assert load_from_gt(tmp_path)[0][30, 30] == 4


@pytest.mark.parametrize('table_one_based', [False, True])
def test_gt_numbering_normalised(tmp_path, table_one_based):
    layout, table = build_sequence()
    if table_one_based:
        table[:, 1:3] += 1
    gt = write_gt(tmp_path, layout, table, start_at=1)
    masks = load_from_gt(gt)
    out, predicted = track_sequence(masks)
    res = tmp_path / 'RES'
    save_result(res, out, predicted)
    gt_out, _ = _prepare(gt, res, tmp_path / 'work')
    _, expected = build_sequence()
    np.testing.assert_array_equal(ctc_io.read_track_table(gt_out / 'TRA/man_track.txt'), expected)


def test_partial_result_rejected(tmp_path):
    layout, table = build_sequence()
    gt = write_gt(tmp_path, layout, table, start_at=0)
    out, predicted = track_sequence({0: frame((1, 30, 30, 4))})
    save_result(tmp_path / 'RES', out, predicted)
    with pytest.raises(ValueError, match='do not score partial'):
        _prepare(gt, tmp_path / 'RES', tmp_path / 'work')


def test_overlap_matrix_sparse_labels():
    a = frame((101, 20, 20, 5), (65535, 40, 40, 5))
    b = frame((97, 21, 20, 5), (3, 40, 41, 5))
    inter = overlap_matrix(a, b, [101, 65535], [3, 97])
    expected = [[np.count_nonzero((a == x) & (b == y)) for y in [3, 97]] for x in [101, 65535]]
    np.testing.assert_array_equal(inter, expected)

"""End-to-end smoke test: synthetic CTC sequence -> tracker -> official metrics.

Run with:  python tests/test_pipeline.py

It builds a 4-frame sequence where one cell moves and then divides, writes it in
the CTC conventions (both 0-based and 1-based numbering), tracks the gold masks
and checks that the result scores SEG = DET = 1.0 and TRA ~ 1.0. That exercises
the file conventions, the tracker and the py-ctcmetrics wiring in one go.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import ctc_io                      # noqa: E402
from src.evaluate import evaluate           # noqa: E402
from src.run_track import save_result       # noqa: E402
from src.track import track_sequence        # noqa: E402

SIZE = 64


def disk(cy: float, cx: float, radius: float) -> np.ndarray:
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2


def build_sequence():
    """frame -> {label: mask} plus the gold lineage."""
    layout = {
        0: {1: disk(20, 20, 6)},
        1: {1: disk(21, 20, 6)},
        2: {2: disk(18, 20, 4), 3: disk(24, 20, 4)},
        3: {2: disk(18, 19, 4), 3: disk(24, 21, 4)},
    }
    table = np.array([[1, 0, 1, 0], [2, 2, 3, 1], [3, 2, 3, 1]], dtype=np.int64)
    return layout, table


def write_gt(root: Path, layout, table, start_at: int) -> Path:
    gt = root / "01_GT"
    for sub, prefix in (("TRA", "man_track"), ("SEG", "man_seg")):
        (gt / sub).mkdir(parents=True, exist_ok=True)
        for frame, labels in layout.items():
            mask = np.zeros((SIZE, SIZE), dtype=np.uint16)
            for label, region in labels.items():
                mask[region] = label
            ctc_io.write_mask(gt / sub / f"{prefix}{frame + start_at:03d}.tif", mask)
    ctc_io.write_track_table(gt / "TRA" / "man_track.txt", table)
    return gt


def run_case(start_at: int) -> dict[str, float]:
    layout, table = build_sequence()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        gt = write_gt(tmp, layout, table, start_at)
        masks = {f: ctc_io.read_mask(p) for f, p in ctc_io.list_masks(gt / "SEG").items()}
        relabelled, predicted = track_sequence(masks, iou_threshold=0.2)
        res = tmp / "01_RES"
        save_result(res, relabelled, predicted)
        problems = ctc_io.check_consistency(
            {f: m for f, m in zip(sorted(masks), relabelled.values())}, predicted
        )
        assert not problems, problems
        return evaluate(gt, res)


def main() -> int:
    for start_at in (0, 1):
        metrics = run_case(start_at)
        label = f"{'0-based' if start_at == 0 else '1-based'} names"
        print(f"{label}: DET={metrics.get('DET'):.4f} SEG={metrics.get('SEG'):.4f} "
              f"TRA={metrics.get('TRA'):.4f}")
        assert metrics["SEG"] > 0.999, metrics
        assert metrics["DET"] > 0.999, metrics
        assert metrics["TRA"] > 0.999, metrics
    print("pipeline: ok (perfect masks + correct lineage -> SEG=DET=TRA=1.0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

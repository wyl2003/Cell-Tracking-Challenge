
"""Evaluate border filtering against GT/TRA using IoU matching."""

from pathlib import Path
import csv

import numpy as np
import tifffile
from scipy.optimize import linear_sum_assignment

from src.segmentation_postprocess import filter_border_cells


ROOT = Path("data/Fluo-N2DH-GOWT1")
PRED_ROOT = Path("runs/cpsam")
MARGINS = [None, 0, 2, 4, 8, 12]
IOU_THRESHOLD = 0.5


def evaluate_frame(pred, gt):
    pred_ids = np.unique(pred)
    pred_ids = pred_ids[pred_ids != 0]

    gt_ids = np.unique(gt)
    gt_ids = gt_ids[gt_ids != 0]

    iou = np.zeros((len(gt_ids), len(pred_ids)))

    for i, g in enumerate(gt_ids):
        gm = gt == g
        for j, p in enumerate(pred_ids):
            pm = pred == p
            intersection = np.logical_and(gm, pm).sum()
            union = np.logical_or(gm, pm).sum()
            iou[i, j] = intersection / union if union else 0

    rows, cols = linear_sum_assignment(-iou)

    tp = sum(
        iou[i, j] >= IOU_THRESHOLD
        for i, j in zip(rows, cols)
    )

    fp = len(pred_ids) - tp
    fn = len(gt_ids) - tp

    return tp, fp, fn


def main():
    results = []

    for seq in ["01", "02"]:
        print(f"\nSequence {seq}")

        for margin in MARGINS:
            total_tp = total_fp = total_fn = 0
            total_removed = 0

            for frame in range(92):
                pred = tifffile.imread(
                    PRED_ROOT / seq / f"mask{frame:03d}.tif"
                )

                gt = tifffile.imread(
                    ROOT / f"{seq}_GT" / "TRA"
                    / f"man_track{frame:03d}.tif"
                )

                if margin is not None:
                    pred, removed = filter_border_cells(
                        pred, margin
                    )
                    total_removed += len(removed)

                tp, fp, fn = evaluate_frame(pred, gt)

                total_tp += tp
                total_fp += fp
                total_fn += fn

            precision = (
                total_tp / (total_tp + total_fp)
                if total_tp + total_fp else 0
            )

            recall = (
                total_tp / (total_tp + total_fn)
                if total_tp + total_fn else 0
            )

            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0
            )

            name = (
                "Baseline" if margin is None
                else f"Border {margin}px"
            )

            row = {
                "sequence": seq,
                "method": name,
                "removed": total_removed,
                "TP": total_tp,
                "FP": total_fp,
                "FN": total_fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "F1": round(f1, 4),
            }

            results.append(row)

            print(
                f"{name:12s} "
                f"TP={total_tp:4d} "
                f"FP={total_fp:4d} "
                f"FN={total_fn:4d} "
                f"F1={f1:.4f}"
            )

    output = Path("runs/border_filter_results.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()

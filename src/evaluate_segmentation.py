
"""Fast instance segmentation evaluation using IoU matching."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def evaluate_masks(pred, gt, threshold=0.5):
    """
    Match predicted cells to ground truth cells.

    Returns TP, FP, FN, precision, recall, F1 and mean IoU.

    Note: These are diagnostic metrics, not official CTC scores.
    """
    if pred.shape != gt.shape:
        raise ValueError("Prediction and GT shapes must match")

    if not (0 <= threshold <= 1):
        raise ValueError("threshold must be between 0 and 1")

    # Relabel both masks to consecutive indices.
    pred_ids, pred_inv = np.unique(
        pred, return_inverse=True
    )
    gt_ids, gt_inv = np.unique(
        gt, return_inverse=True
    )

    # np.bincount requires a 1D array
    pred_inv = pred_inv.ravel()
    gt_inv = gt_inv.ravel()

    n_pred = len(pred_ids)
    n_gt = len(gt_ids)

    # Build an intersection matrix in one pass.
    pairs = gt_inv * n_pred + pred_inv

    intersection = np.bincount(
        pairs,
        minlength=n_gt * n_pred
    ).reshape(n_gt, n_pred)

    gt_area = intersection.sum(axis=1)
    pred_area = intersection.sum(axis=0)

    union = (
        gt_area[:, None]
        + pred_area[None, :]
        - intersection
    )

    iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=float),
        where=union > 0
    )

    # Remove background row and column (label 0).
    gt_fg = gt_ids != 0
    pred_fg = pred_ids != 0

    iou = iou[np.ix_(gt_fg, pred_fg)]

    n_gt_cells = int(gt_fg.sum())
    n_pred_cells = int(pred_fg.sum())

    # Hungarian one-to-one matching.
    if n_gt_cells and n_pred_cells:
        valid = iou >= threshold

        # Prioritize the number of valid matches,
        # then maximize their IoU.
        reward = valid.astype(float) * (
            min(n_gt_cells, n_pred_cells) + 1 + iou
        )

        rows, cols = linear_sum_assignment(-reward)
        matched_iou = [
            float(iou[r, c])
            for r, c in zip(rows, cols)
            if valid[r, c]
        ]
    else:
        matched_iou = []

    tp = len(matched_iou)
    fp = n_pred_cells - tp
    fn = n_gt_cells - tp

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "mean_iou": float(np.mean(matched_iou))
        if matched_iou else 0.0,
    }

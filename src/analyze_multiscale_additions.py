
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path("data/Fluo-N2DH-GOWT1")
BASELINE = Path("runs/cpsam/01")
MULTISCALE = Path("runs/multiscale/seq01_d32")


def best_match(candidate, gt):
    """Find the GT instance with the highest IoU."""
    best_id = 0
    best_iou = 0.0

    candidate_area = np.count_nonzero(candidate)

    for gt_id in np.unique(gt[candidate]):
        if gt_id == 0:
            continue

        gt_region = gt == gt_id

        intersection = np.count_nonzero(candidate & gt_region)
        union = candidate_area + np.count_nonzero(gt_region) - intersection
        iou = intersection / union

        if iou > best_iou:
            best_iou = iou
            best_id = int(gt_id)

    return best_id, best_iou


gt_files = sorted(
    (ROOT / "01_GT" / "SEG").glob("man_seg*.tif")
)

print(
    f"{'Frame':<7} {'New ID':<8} {'Area':<7} "
    f"{'SEG ID':<8} {'SEG IoU':<9} "
    f"{'TRA ID':<8} {'TRA IoU':<9}"
)

total = 0
seg_matches = 0
tra_matches = 0

for gt_path in gt_files:
    frame = gt_path.stem.replace("man_seg", "")

    baseline = tifffile.imread(
        BASELINE / f"mask{frame}.tif"
    )
    merged = tifffile.imread(
        MULTISCALE / f"mask{frame}.tif"
    )

    seg_gt = tifffile.imread(gt_path)
    tra_gt = tifffile.imread(
        ROOT / "01_GT" / "TRA" / f"man_track{frame}.tif"
    )

    # Merge preserves all original IDs and assigns larger IDs
    # to added instances.
    new_ids = np.unique(merged)
    new_ids = new_ids[new_ids > baseline.max()]

    for new_id in new_ids:
        candidate = merged == new_id

        seg_id, seg_iou = best_match(candidate, seg_gt)
        tra_id, tra_iou = best_match(candidate, tra_gt)

        total += 1
        seg_matches += seg_iou >= 0.5
        tra_matches += tra_iou >= 0.5

        print(
            f"{frame:<7} {int(new_id):<8} "
            f"{int(candidate.sum()):<7} "
            f"{seg_id:<8} {seg_iou:<9.4f} "
            f"{tra_id:<8} {tra_iou:<9.4f}"
        )

print("\nSummary:")
print("Added instances:", total)
print("SEG IoU >= 0.5:", seg_matches)
print("TRA IoU >= 0.5:", tra_matches)

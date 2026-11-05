
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path("data/Fluo-N2DH-GOWT1")
GT_DIR = ROOT / "01_GT" / "SEG"
PRED_DIR = Path("runs/cpsam/01")

records = []

for gt_path in sorted(GT_DIR.glob("man_seg*.tif")):
    frame = gt_path.stem.replace("man_seg", "")

    gt = tifffile.imread(gt_path)
    pred = tifffile.imread(PRED_DIR / f"mask{frame}.tif")
    image = tifffile.imread(ROOT / "01" / f"t{frame}.tif")

    height, width = gt.shape

    for gt_id in np.unique(gt):
        if gt_id == 0:
            continue

        gt_region = gt == gt_id
        gt_area = int(gt_region.sum())

        # Find predicted instances overlapping this GT cell.
        candidate_ids = np.unique(pred[gt_region])
        candidate_ids = candidate_ids[candidate_ids != 0]

        best_iou = 0.0
        best_id = 0

        for pred_id in candidate_ids:
            pred_region = pred == pred_id

            intersection = np.count_nonzero(
                gt_region & pred_region
            )
            union = gt_area + pred_region.sum() - intersection

            iou = intersection / union

            if iou > best_iou:
                best_iou = iou
                best_id = int(pred_id)

        ys, xs = np.where(gt_region)

        border_distance = min(
            int(ys.min()),
            int(xs.min()),
            int(height - 1 - ys.max()),
            int(width - 1 - xs.max()),
        )

        mean_brightness = float(image[gt_region].mean())

        records.append({
            "frame": frame,
            "gt_id": int(gt_id),
            "pred_id": best_id,
            "iou": best_iou,
            "area": gt_area,
            "brightness": mean_brightness,
            "border_distance": border_distance,
        })

records.sort(key=lambda x: x["iou"])

print(f"Total GT cells: {len(records)}")
print("\nWorst 15 cells:")
print(
    f"{'Frame':<7} {'GT ID':<7} {'Pred ID':<8} "
    f"{'IoU':<8} {'Area':<9} {'Brightness':<12} {'Border'}"
)

for r in records[:15]:
    print(
        f"{r['frame']:<7} "
        f"{r['gt_id']:<7} "
        f"{r['pred_id']:<8} "
        f"{r['iou']:<8.4f} "
        f"{r['area']:<9} "
        f"{r['brightness']:<12.2f} "
        f"{r['border_distance']}"
    )

print("\nOverall statistics:")
print("Mean best IoU:",
      round(float(np.mean([r["iou"] for r in records])), 4))
print("Median area:",
      round(float(np.median([r["area"] for r in records])), 1))
print("Median brightness:",
      round(float(np.median([r["brightness"] for r in records])), 2))
print("Cells with no overlap:",
      sum(r["pred_id"] == 0 for r in records))

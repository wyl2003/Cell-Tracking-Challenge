
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import center_of_mass

image = tifffile.imread("data/Fluo-N2DH-GOWT1/01/t002.tif")
pred = tifffile.imread("runs/test_frame002/mask000.tif")
gt = tifffile.imread(
    "data/Fluo-N2DH-GOWT1/01_GT/TRA/man_track002.tif"
)

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
matches = [
    (int(gt_ids[i]), int(pred_ids[j]))
    for i, j in zip(rows, cols)
    if iou[i, j] >= 0.5
]

matched_gt = {g for g, p in matches}
matched_pred = {p for g, p in matches}

fp_ids = [int(p) for p in pred_ids if p not in matched_pred]
fn_ids = [int(g) for g in gt_ids if g not in matched_gt]

fig, ax = plt.subplots(figsize=(9, 9))
ax.imshow(image, cmap="gray")

# Red = unmatched predictions
for p in fp_ids:
    region = pred == p
    ax.contour(region, levels=[0.5], colors="red", linewidths=1.5)
    y, x = center_of_mass(region)
    ax.text(x, y, f"FP {p}", color="red", fontsize=9,
            ha="center", va="center", fontweight="bold")

# Green = unmatched ground truth
for g in fn_ids:
    region = gt == g
    ax.contour(region, levels=[0.5], colors="lime", linewidths=1.5)
    y, x = center_of_mass(region)
    ax.text(x, y, f"FN {g}", color="lime", fontsize=9,
            ha="center", va="center", fontweight="bold")

ax.set_title(
    f"Frame 002 - TP={len(matches)}, FP={len(fp_ids)}, FN={len(fn_ids)}"
)
ax.axis("off")
plt.tight_layout()
plt.savefig("runs/test_frame002/error_analysis.png", dpi=180)
plt.show()

print("FP IDs:", fp_ids)
print("FN IDs:", fn_ids)

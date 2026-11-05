
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile

root = Path("data/Fluo-N2DH-GOWT1")
output = Path("runs/diagnostics/missing_cells_overview.png")
output.parent.mkdir(parents=True, exist_ok=True)

cases = [
    ("008", 1),
    ("012", 1),
    ("014", 1),
    ("016", 7),
    ("017", 7),
    ("018", 7),
    ("021", 24),
]

fig, axes = plt.subplots(
    len(cases), 2,
    figsize=(9, 3.5 * len(cases))
)

for row, (frame, gt_id) in enumerate(cases):
    image = tifffile.imread(root / "01" / f"t{frame}.tif")
    gt = tifffile.imread(
        root / "01_GT" / "SEG" / f"man_seg{frame}.tif"
    )
    pred = tifffile.imread(f"runs/cpsam/01/mask{frame}.tif")

    ys, xs = np.where(gt == gt_id)
    cy = int(ys.mean())
    cx = int(xs.mean())

    radius = 90
    y1 = max(0, cy - radius)
    y2 = min(image.shape[0], cy + radius)
    x1 = max(0, cx - radius)
    x2 = min(image.shape[1], cx + radius)

    sl = np.s_[y1:y2, x1:x2]

    axes[row, 0].imshow(image[sl], cmap="gray")
    axes[row, 0].set_title(f"Frame {frame}: Original")

    axes[row, 1].imshow(image[sl], cmap="gray")

    axes[row, 1].contour(
        (gt[sl] == gt_id).astype(float),
        levels=[0.5],
        colors="lime",
        linewidths=1.5,
    )

    axes[row, 1].contour(
        (pred[sl] > 0).astype(float),
        levels=[0.5],
        colors="red",
        linewidths=1,
    )

    axes[row, 1].set_title(
        f"Frame {frame}: GT (green) vs Prediction (red)"
    )

    for ax in axes[row]:
        ax.axis("off")

plt.tight_layout()
plt.savefig(output, dpi=130, bbox_inches="tight")
plt.show()

print("Saved:", output)

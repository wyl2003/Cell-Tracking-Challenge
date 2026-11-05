
from pathlib import Path

import numpy as np
import tifffile
import torch
from cellpose import models

ROOT = Path("data/Fluo-N2DH-GOWT1")
OUTPUT = Path("runs/cellprob_ablation")

THRESHOLDS = [-1.0, -2.0, 1.0]

gt_files = sorted((ROOT / "01_GT" / "SEG").glob("man_seg*.tif"))

model = models.CellposeModel(
    gpu=torch.cuda.is_available(),
    pretrained_model="cpsam_v2",
)

print("CUDA available:", torch.cuda.is_available())
print("Annotated frames:", len(gt_files))

for threshold in THRESHOLDS:
    target = OUTPUT / f"threshold_{threshold}"
    target.mkdir(parents=True, exist_ok=True)

    print(f"\nTesting cellprob_threshold={threshold}", flush=True)

    for i, gt_path in enumerate(gt_files, start=1):
        frame = gt_path.stem.replace("man_seg", "")

        img_path = ROOT / "01" / f"t{frame}.tif"
        img = tifffile.imread(img_path)

        masks, _, _ = model.eval(
            img,
            diameter=48,
            channels=[0, 0],
            flow_threshold=0.4,
            cellprob_threshold=threshold,
            min_size=200,
        )

        tifffile.imwrite(
            target / f"mask{frame}.tif",
            masks.astype(np.uint16),
        )

        print(
            f"[{i}/{len(gt_files)}] Frame {frame}: "
            f"{len(np.unique(masks)) - 1} cells",
            flush=True,
        )

print("\nAll experiments completed.")


from pathlib import Path
import shutil

import numpy as np
import tifffile
import torch
from cellpose import models

from src.segmentation_postprocess import merge_multiscale_masks

ROOT = Path("data/Fluo-N2DH-GOWT1")
BASELINE = Path("runs/cpsam/01")
OUTPUT = Path("runs/multiscale/seq01_d32")

GT_DIR = ROOT / "01_GT" / "SEG"
image_files = sorted((ROOT / "01").glob("t*.tif"))

OUTPUT = Path("runs/multiscale/seq01_d32_full")
OUTPUT.mkdir(parents=True, exist_ok=True)

# Copy all 92 baseline masks to create a complete evaluation sequence.
baseline_files = sorted(BASELINE.glob("mask*.tif"))
assert len(baseline_files) == 92

for path in baseline_files:
    shutil.copy2(path, OUTPUT / path.name)

model = models.CellposeModel(
    gpu=torch.cuda.is_available(),
    pretrained_model="cpsam_v2",
)

total_added = 0

print("CUDA:", torch.cuda.is_available())
print("Total frames:", len(image_files))

for i, image_path in enumerate(image_files, start=1):
    frame = image_path.stem.replace("t", "")

    image = tifffile.imread(image_path)

    baseline = tifffile.imread(
        BASELINE / f"mask{frame}.tif"
    )

    small, _, _ = model.eval(
        image,
        diameter=32,
        channels=[0, 0],
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        min_size=200,
    )

    merged = merge_multiscale_masks(
        baseline,
        small,
        overlap_threshold=0.10,
        min_area=200,
    )

    before = np.count_nonzero(np.unique(baseline))
    after = np.count_nonzero(np.unique(merged))
    added = after - before

    total_added += added

    # Verify that existing baseline pixels remain unchanged.
    assert np.array_equal(
        merged[baseline > 0],
        baseline[baseline > 0],
    )

    tifffile.imwrite(
        OUTPUT / f"mask{frame}.tif",
        merged.astype(np.uint16),
    )

    print(
        f"[{i:02d}/{len(image_files)}] "
        f"Frame {frame}: "
        f"{before} -> {after} cells (+{added})",
        flush=True,
    )

print("\nTotal added instances:", total_added)
print("Output masks:", len(list(OUTPUT.glob("mask*.tif"))))
print("Saved to:", OUTPUT)

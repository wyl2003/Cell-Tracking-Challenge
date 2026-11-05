
from pathlib import Path

import numpy as np
import tifffile
import torch
from cellpose import models

from src.segmentation_postprocess import merge_multiscale_masks

ROOT = Path("data/Fluo-N2DH-GOWT1")
BASELINE = Path("runs/cpsam/02")
OUTPUT = Path("runs/multiscale/seq02_d32")

OUTPUT.mkdir(parents=True, exist_ok=True)

baseline_files = sorted(BASELINE.glob("mask*.tif"))
assert len(baseline_files) == 92, "Expected 92 baseline frames"

model = models.CellposeModel(
    gpu=torch.cuda.is_available(),
    pretrained_model="cpsam_v2",
)

total_added = 0
changed_frames = 0

print("CUDA available:", torch.cuda.is_available())
print("Frames:", len(baseline_files))

for i, path in enumerate(baseline_files, start=1):
    frame = path.stem.replace("mask", "")

    image = tifffile.imread(ROOT / "02" / f"t{frame}.tif")
    baseline = tifffile.imread(path)

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

    # Confirm that the original cells are preserved.
    assert np.array_equal(
        merged[baseline > 0],
        baseline[baseline > 0],
    )

    before = np.count_nonzero(np.unique(baseline))
    after = np.count_nonzero(np.unique(merged))
    added = after - before

    total_added += added
    changed_frames += added > 0

    tifffile.imwrite(
        OUTPUT / path.name,
        merged.astype(np.uint16),
    )

    print(
        f"[{i:02d}/92] Frame {frame}: "
        f"{before} -> {after} cells (+{added})",
        flush=True,
    )

print("\nTotal added instances:", total_added)
print("Frames with additions:", changed_frames)
print("Output masks:", len(list(OUTPUT.glob("mask*.tif"))))
print("Saved to:", OUTPUT)

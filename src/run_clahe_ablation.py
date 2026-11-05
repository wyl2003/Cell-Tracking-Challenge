
"""CLAHE ablation study on Sequence 01."""

from pathlib import Path
import csv

import cv2
import numpy as np
import tifffile
import torch

from cellpose import models

from src.evaluate_segmentation import evaluate_masks

ROOT = Path("data/Fluo-N2DH-GOWT1")
BASELINE = Path("runs/cpsam/01")
OUTPUT = Path("runs/clahe_ablation")

CLIP_LIMITS = [1.0, 2.0, 3.0]
MIN_SIZE = 200


def apply_clahe(image, clip_limit):
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(8, 8)
    )
    return clahe.apply(image)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    gt_files = sorted(
        (ROOT / "01_GT" / "SEG").glob("man_seg*.tif")
    )

    print(f"Evaluating {len(gt_files)} annotated frames")

    model = models.CellposeModel(
        gpu=torch.cuda.is_available(),
        pretrained_model="cpsam_v2"
    )

    results = []

    for clip in CLIP_LIMITS:
        print(f"\nCLAHE clipLimit={clip}")

        clip_dir = OUTPUT / f"clip_{clip}"
        clip_dir.mkdir(parents=True, exist_ok=True)

        total_tp = total_fp = total_fn = 0

        for path in gt_files:
            frame = path.stem.replace("man_seg", "")

            image = tifffile.imread(
                ROOT / "01" / f"t{frame}.tif"
            )

            enhanced = apply_clahe(image, clip)

            masks, _, _ = model.eval(
                enhanced,
                diameter=48,
                channels=[0, 0],
                flow_threshold=0.4,
                cellprob_threshold=0.0,
                min_size=MIN_SIZE
            )

            masks = masks.astype(np.uint16)

            tifffile.imwrite(
                clip_dir / f"mask{frame}.tif",
                masks,
                compression="zlib"
            )

            # Use TRA annotation for detection analysis.
            gt = tifffile.imread(
                ROOT / "01_GT" / "TRA"
                / f"man_track{frame}.tif"
            )

            result = evaluate_masks(masks, gt)

            total_tp += result["TP"]
            total_fp += result["FP"]
            total_fn += result["FN"]

            print(
                f"Frame {frame}: "
                f"TP={result['TP']} "
                f"FP={result['FP']} "
                f"FN={result['FN']}"
            )

        precision = total_tp / (total_tp + total_fp)
        recall = total_tp / (total_tp + total_fn)
        f1 = 2 * precision * recall / (precision + recall)

        results.append({
            "clip_limit": clip,
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "precision": precision,
            "recall": recall,
            "F1": f1
        })

    with (OUTPUT / "results.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nSummary:")
    for result in results:
        print(
            f"clip={result['clip_limit']}: "
            f"TP={result['TP']}, "
            f"FP={result['FP']}, "
            f"FN={result['FN']}, "
            f"F1={result['F1']:.4f}"
        )


if __name__ == "__main__":
    main()

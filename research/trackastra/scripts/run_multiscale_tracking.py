"""Compare tracking configurations on labeled data or export blind test predictions."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import time
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import numpy as np
import tifffile

from curate_results import sha256, write_csv
from run_finetune_experiments import find_ctc_evaluate, run_command, write_json


BASE_DIAMETER = 48
SMALL_DIAMETER = 32
OVERLAP_THRESHOLD = 0.10
MIN_AREA = 200
CELLPOSE_OPTIONS = {"flow_threshold": 0.4, "cellprob_threshold": 0.0, "min_size": MIN_AREA}
METRICS = (
    "DET", "SEG", "TRA", "CT", "TF", "OP_CTB", "AOGM",
    "BC(0)", "gt_divisions", "tp_div(0)", "fp_div(0)", "fn_div(0)",
)


def segment(frames, folders, merge_source, weights, device):
    import torch
    from cellpose import models

    spec = importlib.util.spec_from_file_location("team_segmentation_postprocess", merge_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    model = None
    rows = []
    for index, frame in enumerate(frames):
        name = f"mask{index:03d}.tif"
        baseline_file = folders["single_scale"] / name
        merged_file = folders["multiscale"] / name
        if baseline_file.is_file() and merged_file.is_file():
            baseline = tifffile.imread(baseline_file)
            merged = tifffile.imread(merged_file)
        else:
            if model is None:
                print(f"Loading Cellpose-SAM v2 on {device}", flush=True)
                model = models.CellposeModel(device=torch.device(device), pretrained_model=str(weights))
            image = tifffile.imread(frame)
            if baseline_file.is_file():
                baseline = tifffile.imread(baseline_file)
            else:
                baseline, _, _ = model.eval(image, diameter=BASE_DIAMETER, **CELLPOSE_OPTIONS)
                write_mask(baseline_file, baseline)
            small, _, _ = model.eval(image, diameter=SMALL_DIAMETER, **CELLPOSE_OPTIONS)
            merged = module.merge_multiscale_masks(
                baseline, small, overlap_threshold=OVERLAP_THRESHOLD, min_area=MIN_AREA,
            )
            write_mask(merged_file, merged)
        if baseline.shape != merged.shape or not np.array_equal(merged[baseline > 0], baseline[baseline > 0]):
            raise ValueError(f"Multi-scale mask changed existing detections in {name}")
        before = int(np.count_nonzero(np.unique(baseline)))
        after = int(np.count_nonzero(np.unique(merged)))
        rows.append({"frame": index, "single_scale": before, "multiscale": after, "added": after - before})
        if index == 0 or (index + 1) % 10 == 0 or index + 1 == len(frames):
            print(f"SEGMENT {index + 1}/{len(frames)} frames", flush=True)
    return rows


def write_mask(path: Path, mask: np.ndarray) -> None:
    if mask.ndim != 2 or mask.min() < 0 or mask.max() > np.iinfo(np.uint16).max:
        raise ValueError(f"Mask cannot be stored as 2D uint16: {path}")
    temporary = path.with_suffix(".tmp")
    tifffile.imwrite(temporary, mask.astype(np.uint16), compression="zlib")
    temporary.replace(path)


def export_tracks(model_dir, mode, images, masks, result_dir, tracking_file, device, frame_count, digits):
    from trackastra.model import Trackastra
    from trackastra.tracking import graph_to_ctc

    result_dir.parent.mkdir(parents=True, exist_ok=True)
    expected_masks = {f"mask{i:0{digits}d}.tif" for i in range(frame_count)}
    if not tracking_file.is_file():
        if result_dir.exists():
            raise FileExistsError(f"Existing result without tracking metadata: {result_dir}")
        start = time.perf_counter()
        model = Trackastra.from_folder(model_dir, device=device, batch_size=1)
        graph, tracked_masks = model.track_from_disk(images, masks, mode=mode)
        staged = result_dir.with_name(f"{result_dir.name}.partial_{uuid4().hex[:8]}")
        graph_to_ctc(graph, tracked_masks, outdir=staged)
        (staged / "man_track.txt").rename(staged / "res_track.txt")
        for i in range(frame_count):
            (staged / f"man_track{i:04d}.tif").rename(staged / f"mask{i:0{digits}d}.tif")
        staged.rename(result_dir)
        write_json(tracking_file, {
            "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
            "predicted_splits": sum(degree == 2 for _, degree in graph.out_degree()),
            "track_and_export_s": round(time.perf_counter() - start, 3),
        })
    if ({path.name for path in result_dir.glob("*.tif")} != expected_masks
            or not (result_dir / "res_track.txt").is_file()):
        raise ValueError(f"CTC result does not cover the input frames: {result_dir}")
    return json.loads(tracking_file.read_text(encoding="utf-8"))


def track_and_score(model_dir, mode, images, masks, gt, run_dir, device, evaluator, threads, frame_count):
    result_dir = run_dir / f"{images.name}_RES"
    tracking = export_tracks(model_dir, mode, images, masks, result_dir,
                             run_dir / "tracking.json", device, frame_count, digits=4)
    metrics_file = run_dir / "ctc_metrics.csv"
    if not metrics_file.is_file():
        partial = metrics_file.with_suffix(".partial.csv")
        run_command([evaluator, "--res", str(result_dir), "--gt", str(gt),
                     "--det", "--seg", "--tra", "--ct", "--tf", "--bc", "1",
                     "-n", str(threads), "--csv-file", str(partial)], run_dir / "metrics.log")
        partial.rename(metrics_file)
    with metrics_file.open(encoding="utf-8-sig", newline="") as stream:
        scores, = csv.DictReader(stream, delimiter=";")
    for key in ("DET", "SEG", "TRA", "CT", "AOGM"):
        float(scores[key])
    return {**tracking, **{key: scores.get(key, "") for key in METRICS}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Fluo-N2DH-GOWT1 folder")
    parser.add_argument("--sequence", choices=("01", "02"), default="02")
    parser.add_argument("--segmentation-source", type=Path, required=True,
                        help="Teammate's src/segmentation_postprocess.py")
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--finetuned", type=Path, help="Required when comparing validation scores")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predict-only", action="store_true",
                        help="Export <sequence>_RES using Multi-scale + general_2d + ILP without ground truth")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--ctc-evaluate")
    parser.add_argument("--eval-threads", type=int, default=4)
    args = parser.parse_args()
    if args.eval_threads < 1:
        parser.error("--eval-threads must be positive")
    if not args.predict_only and args.finetuned is None:
        parser.error("--finetuned is required for the validation comparison")

    os.environ.setdefault("TQDM_DISABLE", "1")
    import torch
    from cellpose import models

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this Python environment")
    images = args.dataset.resolve() / args.sequence
    frames = sorted(images.glob("t*.tif"))
    if not frames:
        parser.error(f"No t*.tif images in {images}")
    digits = len(frames[0].stem) - 1
    if digits < 3 or [path.name for path in frames] != [f"t{i:0{digits}d}.tif" for i in range(len(frames))]:
        parser.error(f"Expected consecutive zero-based CTC image files in {images}")

    gt = None if args.predict_only else images.with_name(f"{args.sequence}_GT")
    if gt is not None and (
        not (gt / "TRA/man_track.txt").is_file()
        or not all((gt / "TRA" / f"man_track{i:0{digits}d}.tif").is_file() for i in range(len(frames)))
        or not (gt / "SEG").is_dir()
    ):
        parser.error("Validation comparison requires matching TRA annotations and SEG ground truth")
    weights = Path(models.cache_model_path("cpsam_v2")).resolve()
    model_dirs = {"general_2d": args.pretrained.resolve()}
    if not args.predict_only:
        model_dirs["finetuned"] = args.finetuned.resolve()
    for folder in model_dirs.values():
        for filename in ("model.pt", "config.yaml", "train_config.yaml"):
            if not (folder / filename).is_file():
                raise FileNotFoundError(folder / filename)
    merge_source = args.segmentation_source.resolve()
    evaluator = None if args.predict_only else find_ctc_evaluate(args.ctc_evaluate)
    output = args.output.resolve()
    root = output / "work" / args.sequence if args.predict_only else output
    root.mkdir(parents=True, exist_ok=True)
    run = {
        "images": str(images), "ground_truth": str(gt) if gt is not None else None, "device": args.device,
        "cellpose": version("cellpose"), "trackastra": version("trackastra"), "torch": version("torch"),
        "cellpose_model": "cpsam_v2", "cellpose_sha256": sha256(weights),
        "diameters": [BASE_DIAMETER, SMALL_DIAMETER], "cellpose_options": CELLPOSE_OPTIONS,
        "merge_source": str(merge_source), "merge_source_sha256": sha256(merge_source),
        "merge_overlap_threshold": OVERLAP_THRESHOLD,
        "tracking_model_sha256": {name: sha256(path / "model.pt") for name, path in model_dirs.items()},
        "evaluator": evaluator,
    }
    if args.predict_only:
        run.update(frame_count=len(frames), segmentation="multiscale", tracking_mode="ilp")
    run_file = root / "run.json"
    if run_file.exists():
        if json.loads(run_file.read_text(encoding="utf-8")) != run:
            raise ValueError("Existing experiment has different inputs. Use a new --output directory.")
    elif any(root.iterdir()):
        raise FileExistsError(f"Output directory already contains files: {root}")
    else:
        write_json(run_file, run)

    folders = {name: root / "masks" / name for name in ("single_scale", "multiscale")}
    frame_counts = segment(frames, folders, merge_source, weights, args.device)
    write_csv(root / "segmentation_counts.csv", list(frame_counts[0]), frame_counts)
    print(f"Multi-scale added {sum(row['added'] for row in frame_counts)} instances", flush=True)
    if args.device == "cuda":
        torch.cuda.empty_cache()

    if args.predict_only:
        result_dir = output / f"{args.sequence}_RES"
        print(f"TRACK {args.sequence} / multiscale / general_2d / ilp", flush=True)
        export_tracks(model_dirs["general_2d"], "ilp", images, folders["multiscale"],
                      result_dir, root / "tracking.json", args.device, len(frames), digits)
        print(f"EXPORTED {len(frames)} frames to {result_dir}", flush=True)
        return

    results = []
    for mask_name, masks in folders.items():
        for model_name, model_dir in model_dirs.items():
            for mode in ("greedy", "ilp"):
                print(f"TRACK {mask_name} / {model_name} / {mode}", flush=True)
                scores = track_and_score(model_dir, mode, images, masks, gt,
                                         root / mask_name / f"{model_name}_{mode}",
                                         args.device, evaluator, args.eval_threads, len(frames))
                results.append({"segmentation": mask_name, "model": model_name, "mode": mode, **scores})
                write_csv(root / "comparison.csv", list(results[0]), results)
                print(f"RESULT TRA={scores['TRA']} CT={scores['CT']} OP_CTB={scores['OP_CTB']}", flush=True)


if __name__ == "__main__":
    main()

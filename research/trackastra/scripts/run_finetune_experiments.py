"""Repeatable Trackastra fine-tuning and 02 validation-sequence evaluation.

Run with the same Python environment that was used for scripts/train.py.
The input YAML and the official Trackastra source files are never edited.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml


SCRIPTS_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPTS_DIR.parent
PROJECT_DIR = SCRIPTS_DIR.parents[2]
DEFAULT_CONFIG = RESEARCH_DIR / "configs" / "finetune_config.example.yaml"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "trackastra" / "finetune"
RESULT_COLUMNS = [
    "experiment", "status", "epochs_requested", "actual_epochs", "best_epoch",
    "train_samples", "lr", "seed", "tracking_mode", "best_val_loss", "CT", "TRA", "AOGM",
    "evaluation_set", "pretrained_model", "best_model_dir", "ctc_result_dir",
    "config_file", "error",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, data: dict) -> None:
    """Atomically update a script-owned manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def read_base_config(path: Path) -> tuple[dict, Path, Path, Path]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML mapping: {path}")
    model = Path(config.get("model", "")).expanduser().resolve()
    if model.name != "general_2d" or not all(
        (model / name).is_file() for name in ("model.pt", "config.yaml", "train_config.yaml")
    ):
        raise ValueError("model must point to the installed general_2d pretrained folder")
    inputs = config.get("input_val")
    if not isinstance(inputs, list) or len(inputs) != 1:
        raise ValueError("input_val must contain exactly one fixed validation sequence")
    sequence = Path(inputs[0]).expanduser().resolve()
    if sequence.name != "02" or not sequence.is_dir():
        raise ValueError(f"Expected the existing 02 validation sequence: {sequence}")
    gt = sequence.parent / "02_GT"
    tra = gt / "TRA"
    if not (tra / "man_track.txt").is_file() or not list(tra.glob("man_track*.tif")):
        raise ValueError(f"Missing 02 GT/TRA annotations: {tra}")
    if not list(sequence.glob("*.tif")):
        raise ValueError(f"No 02 image frames in {sequence}")
    if not config.get("input_train"):
        raise ValueError("input_train is required")
    for item in config["input_train"]:
        if not Path(item).is_dir():
            raise ValueError(f"Missing training sequence: {item}")
    return config, model, sequence, gt


def experiment_name(config_path: Path, config: dict, epochs: int, samples: int,
                    lr: float, seed: int, mode: str) -> str:
    identity = {
        "config_path": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "epochs": epochs, "train_samples": samples, "lr": lr, "seed": seed,
        "mode": mode,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    return f"e{epochs}_s{samples}_lr{lr:g}_{digest}"


def make_training_config(base: dict, model: Path, attempt_dir: Path,
                         epochs: int, samples: int, lr: float, seed: int,
                         cache_dir: Path | None = None) -> dict:
    """Each attempt starts from the same source checkpoint in a new directory."""
    config = dict(base)
    config.update({
        "model": str(model), "epochs": epochs, "train_samples": samples,
        "lr": lr, "seed": seed, "outdir": str(attempt_dir), "name": "train",
        "timestamp": False, "resume": False, "logger": "tensorboard",
    })
    if cache_dir is not None:
        config["cachedir"] = str(cache_dir)
    return config


def best_validation(run_dir: Path) -> tuple[int, int, float]:
    """Read actual validation epochs and losses from the official TensorBoard log."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    versions = sorted((run_dir / "tb").glob("version_*"))
    if len(versions) != 1:
        raise RuntimeError(f"Expected one TensorBoard version in {run_dir / 'tb'}")
    events = EventAccumulator(str(versions[0]))
    events.Reload()
    if "val_loss" not in events.Tags()["scalars"]:
        raise RuntimeError(f"No val_loss events in {versions[0]}")
    # The final value for any repeated global step is authoritative.
    losses = {event.step: float(event.value) for event in events.Scalars("val_loss")}
    ordered = sorted(losses.items())
    if not ordered or not all(math.isfinite(loss) for _, loss in ordered):
        raise RuntimeError("Missing or nonfinite validation loss")
    best_index = min(range(len(ordered)), key=lambda index: ordered[index][1])
    return len(ordered), best_index + 1, ordered[best_index][1]


def run_command(command: list[str], log_file: Path, cwd: Path | None = None) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    # Lightning/Rich prints Unicode progress markers on Windows. The inherited
    # GBK encoding can otherwise turn a successful training run into an error.
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"
    with log_file.open("w", encoding="utf-8", errors="replace") as stream:
        stream.write("Command: " + subprocess.list2cmdline(command) + "\n\n")
        stream.flush()
        result = subprocess.run(command, cwd=cwd, stdout=stream,
                                stderr=subprocess.STDOUT, check=False,
                                env=child_env)
    if result.returncode:
        raise RuntimeError(f"Exit code {result.returncode}; see {log_file}")


def find_ctc_evaluate(override: str | None) -> str:
    if override:
        executable = Path(override).expanduser()
        if not executable.is_file():
            raise FileNotFoundError(f"ctc_evaluate not found: {executable}")
        return str(executable.resolve())
    on_path = shutil.which("ctc_evaluate")
    if on_path:
        return on_path
    # The local project uses conda envs/cell for training and envs/ctc-eval for metrics.
    sibling = Path(sys.executable).resolve().parent.parent / "ctc-eval" / "Scripts" / "ctc_evaluate.exe"
    if sibling.is_file():
        return str(sibling)
    raise FileNotFoundError("ctc_evaluate not found; pass --ctc-evaluate PATH")


def export_ctc(run_dir: Path, sequence: Path, gt: Path, output: Path,
               device: str, mode: str) -> None:
    """Use the best model.pt and the same GT/TRA detections as the manual test."""
    from trackastra.model import Trackastra
    from trackastra.tracking import graph_to_ctc

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite CTC results: {output}")
    model = Trackastra.from_folder(run_dir, device=device, batch_size=1)
    graph, masks = model.track_from_disk(
        imgs_path=sequence, masks_path=gt / "TRA", mode=mode,
    )
    graph_to_ctc(graph, masks, outdir=output)
    (output / "man_track.txt").rename(output / "res_track.txt")
    for path in output.glob("man_track*.tif"):
        path.rename(output / path.name.replace("man_track", "mask", 1))
    validate_ctc(output, gt)


def validate_ctc(output: Path, gt: Path) -> None:
    if not (output / "res_track.txt").is_file():
        raise RuntimeError(f"Missing res_track.txt in {output}")
    expected = len(list((gt / "TRA").glob("man_track*.tif")))
    actual = len(list(output.glob("mask*.tif")))
    if expected != actual:
        raise RuntimeError(f"CTC mask count {actual} differs from GT count {expected}")


def read_metrics(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter=";"))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one metrics row in {path}")
    metrics = {key: float(rows[0][key]) for key in ("TRA", "CT", "AOGM")}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError(f"Nonfinite metrics in {path}")
    return metrics


def collect_results(output_root: Path) -> Path:
    rows = []
    for manifest in output_root.glob("*/result.json"):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            rows.append({key: data.get(key, "") for key in RESULT_COLUMNS})
        except (OSError, ValueError):
            continue
    def write_ranked(filename: str, primary: str, secondary: str) -> Path:
        ranked = sorted(rows, key=lambda row: (
            row["status"] != "complete",
            -float(row[primary] or "-inf"), -float(row[secondary] or "-inf"),
            row["experiment"],
        ))
        target = output_root / filename
        temporary = output_root / (filename + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            writer.writerows(ranked)
        os.replace(temporary, target)
        return target

    target = write_ranked("results.csv", "CT", "TRA")
    write_ranked("results_by_ct.csv", "CT", "TRA")
    write_ranked("results_by_tra.csv", "TRA", "CT")
    return target


def run_experiment(args: argparse.Namespace, base: dict, model: Path,
                   sequence: Path, gt: Path, epochs: int, samples: int,
                   lr: float, ctc_evaluate: str) -> dict:
    name = experiment_name(args.config, base, epochs, samples, lr, args.seed, args.mode)
    root = args.output_root / name
    manifest = root / "result.json"
    if manifest.is_file():
        try:
            prior = json.loads(manifest.read_text(encoding="utf-8"))
            if prior.get("status") == "complete":
                validate_ctc(Path(prior["ctc_result_dir"]), gt)
                if Path(prior["best_model_dir"], "model.pt").is_file():
                    if "tracking_mode" not in prior:
                        prior["tracking_mode"] = args.mode
                        write_json(manifest, prior)
                    print(f"SKIP complete: {name}", flush=True)
                    return prior
        except (KeyError, OSError, ValueError, RuntimeError) as exc:
            print(f"RECHECK incomplete result {name}: {exc}", flush=True)
    root.mkdir(parents=True, exist_ok=True)
    row = {
        "experiment": name, "status": "failed", "epochs_requested": epochs,
        "actual_epochs": "", "best_epoch": "", "train_samples": samples,
        "lr": lr, "seed": args.seed, "tracking_mode": args.mode,
        "best_val_loss": "", "CT": "",
        "TRA": "", "AOGM": "", "evaluation_set": "02 validation (not independent test)",
        "pretrained_model": str(model), "best_model_dir": "",
        "ctc_result_dir": str(root / "02_RES"), "config_file": "", "error": "",
    }
    trained = sorted(root.glob("attempt_*/training.json"))
    completed_training = None
    trained_now = False
    for path in reversed(trained):
        try:
            candidate = json.loads(path.read_text(encoding="utf-8"))
            if Path(candidate["best_model_dir"], "model.pt").is_file():
                completed_training = candidate
                break
        except (KeyError, OSError, ValueError):
            continue
    try:
        for retry in range(args.max_retries + 1):
            try:
                if completed_training is None:
                    existing = [int(p.name.split("_")[-1]) for p in root.glob("attempt_*" )
                                if p.is_dir() and p.name.split("_")[-1].isdigit()]
                    number = max(existing, default=0) + 1
                    attempt = root / f"attempt_{number:02d}"
                    attempt.mkdir()
                    config = make_training_config(base, model, attempt, epochs, samples, lr,
                                                  args.seed, args.cache_dir)
                    config_file = attempt / "config.yaml"
                    with config_file.open("w", encoding="utf-8") as stream:
                        yaml.safe_dump(config, stream, sort_keys=False)
                    run_dir = attempt / "train"
                    print(f"TRAIN {name}, attempt {number}", flush=True)
                    run_command([args.python, str(args.trackastra_root / "scripts" / "train.py"),
                                 "-c", str(config_file)], attempt / "train.log",
                                cwd=args.trackastra_root)
                    if not all((run_dir / filename).is_file() for filename in
                               ("model.pt", "config.yaml", "train_config.yaml")):
                        raise RuntimeError(f"Best model files missing in {run_dir}")
                    actual, best_epoch, best_loss = best_validation(run_dir)
                    completed_training = {
                        "best_model_dir": str(run_dir), "config_file": str(config_file),
                        "actual_epochs": actual, "best_epoch": best_epoch,
                        "best_val_loss": best_loss,
                    }
                    write_json(attempt / "training.json", completed_training)
                    trained_now = True
                row.update(completed_training)
                final_res = root / "02_RES"
                if final_res.exists():
                    try:
                        if trained_now:
                            raise RuntimeError("A new model was trained for this result")
                        validate_ctc(final_res, gt)
                    except RuntimeError:
                        # Preserve unexpected contents rather than overwrite them.
                        archived = root / f"02_RES.incomplete_{uuid.uuid4().hex[:8]}"
                        final_res.rename(archived)
                if not final_res.exists():
                    staging = root / "02_RES.partial"
                    if staging.exists():
                        shutil.rmtree(staging)
                    print(f"INFER {name} on 02 validation", flush=True)
                    export_ctc(Path(completed_training["best_model_dir"]), sequence,
                               gt, staging, args.device, args.mode)
                    staging.rename(final_res)
                validate_ctc(final_res, gt)
                metrics_file = root / "ctc_metrics.csv"
                print(f"METRICS {name}", flush=True)
                run_command([ctc_evaluate, "--res", str(final_res), "--gt", str(gt),
                             "--tra", "--ct", "-n", "1", "--csv-file",
                             str(metrics_file)], root / f"metrics_attempt_{retry + 1:02d}.log")
                row.update(read_metrics(metrics_file))
                row.update(status="complete", error="")
                write_json(manifest, row)
                return row
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                with (root / "errors.log").open("a", encoding="utf-8") as stream:
                    stream.write(f"[{now()}] retry {retry + 1}\n{traceback.format_exc()}\n")
                print(f"FAILED {name}, retry {retry + 1}: {row['error']}", flush=True)
        write_json(manifest, row)
        return row
    except BaseException:
        write_json(manifest, row)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--trackastra-root", type=Path,
                        help="Checkout containing official scripts/train.py")
    parser.add_argument("--cache-dir", type=Path,
                        help="Pin the official training cache; default: <trackastra-root>/.cache")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, nargs="+", help="Epoch grid")
    parser.add_argument("--train-samples", type=int, nargs="+", help="Sample grid")
    parser.add_argument("--lrs", type=float, nargs="+", help="Learning-rate grid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("greedy", "greedy_nodiv", "ilp"), default="greedy")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--python", default=sys.executable, help="Training Python executable")
    parser.add_argument("--ctc-evaluate", help="Path to py-ctcmetrics ctc_evaluate")
    parser.add_argument("--max-retries", type=int, default=1, help="Retries after first failure")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.config = args.config.resolve()
    args.output_root = args.output_root.resolve()
    if args.trackastra_root is not None:
        args.trackastra_root = args.trackastra_root.resolve()
        if not (args.trackastra_root / "scripts" / "train.py").is_file():
            parser.error("--trackastra-root must contain scripts/train.py")
    if not args.dry_run and args.trackastra_root is None:
        parser.error("--trackastra-root is required unless --dry-run")
    args.cache_dir = (args.cache_dir.resolve() if args.cache_dir is not None else
                      args.trackastra_root / ".cache" if args.trackastra_root else None)
    if args.max_retries < 0:
        parser.error("--max-retries must be nonnegative")
    return args


def main(argv: list[str] | None = None) -> int:
    # tqdm reads this at import time; keep batch runs readable in terminals.
    os.environ.setdefault("TQDM_DISABLE", "1")
    args = parse_args(argv)
    base, model, sequence, gt = read_base_config(args.config)
    base["model"] = str(model)
    base["input_train"] = [str(Path(item).expanduser().resolve()) for item in base["input_train"]]
    base["input_val"] = [str(sequence)]
    epochs_grid = args.epochs or [int(base["epochs"])]
    samples_grid = args.train_samples or [int(base["train_samples"])]
    lr_grid = args.lrs or [float(base["lr"])]
    if any(x <= 0 for x in epochs_grid + samples_grid) or any(
        not math.isfinite(x) or x <= 0 for x in lr_grid
    ):
        raise ValueError("epochs, train_samples and lrs must be positive")
    experiments = list(dict.fromkeys(itertools.product(epochs_grid, samples_grid, lr_grid)))
    print(f"{len(experiments)} experiment(s); 02 is a validation sequence, not an independent test set")
    if args.dry_run:
        for epochs, samples, lr in experiments:
            print(experiment_name(args.config, base, epochs, samples, lr, args.seed, args.mode))
        return 0
    ctc_evaluate = find_ctc_evaluate(args.ctc_evaluate)
    args.output_root.mkdir(parents=True, exist_ok=True)
    failed = 0
    for epochs, samples, lr in experiments:
        result = run_experiment(args, base, model, sequence, gt,
                                epochs, samples, lr, ctc_evaluate)
        failed += result["status"] != "complete"
        print(f"Results: {collect_results(args.output_root)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

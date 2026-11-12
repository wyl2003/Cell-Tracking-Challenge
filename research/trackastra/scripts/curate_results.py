"""Export small, path-free Trackastra research artifacts from a local run.

Only CSV/JSON metadata and graph tables are copied. Model weights, TIFF images,
CTC masks, caches, logs and virtual environments are deliberately excluded.
This command reads completed local experiments; it never starts training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent / "results"
RUN_COLUMNS = ["experiment", "status", "epochs_requested", "actual_epochs",
               "best_epoch", "train_samples", "lr", "seed", "tracking_mode",
               "best_val_loss", "CT", "TRA", "AOGM", "evaluation_set",
               "checkpoint_sha256"]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def safe_runs(source: Path, output: Path) -> list[dict]:
    rows = read_csv(source)
    if not rows or any(row["status"] != "complete" for row in rows):
        raise ValueError(f"Expected completed experiments: {source}")
    result = []
    for row in rows:
        checkpoint = Path(row["best_model_dir"]) / "model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        result.append({key: row.get(key, "") for key in RUN_COLUMNS}
                      | {"checkpoint_sha256": sha256(checkpoint)})
    write_csv(output, RUN_COLUMNS, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-scripts", type=Path, required=True,
                        help="scripts/ in the completed local Trackastra checkout")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    source, out = args.local_scripts.resolve(), args.out.resolve()
    grid = safe_runs(source / "finetune_sweep_01/results.csv", out / "grid_search.csv")
    seeds = safe_runs(source / "finetune_seed_repeats/results.csv", out / "seed_repeats.csv")
    if len(grid) != 8 or len(seeds) != 3:
        raise ValueError("Expected eight grid runs and three seed runs")

    val = source / "finetune_validation_02"
    mode = source / "greedy_vs_ilp_02"
    controlled = read_csv(source / "finetune_cache_reproduction/results.csv")
    if len(controlled) != 1 or controlled[0]["status"] != "complete":
        raise ValueError("Expected one controlled cache reproduction")
    prior = json_file(val / "provenance.json")
    manual = json_file(val / "manual_exp2.json")
    root_cache = next(row for row in seeds if row["seed"] == "20839")
    cache_rows = [
        {"condition": "manual_exp2", "seed": 20839,
         "actual_epochs": manual["actual_epochs"], "best_epoch": manual["best_epoch"],
         "best_val_loss": manual["best_val_loss"], "CT": manual["CT"],
         "TRA": manual["TRA"], "AOGM": manual["AOGM"],
         "checkpoint_sha256": prior["checkpoint_sha256"]["manual_exp2"]},
        {"condition": "same_seed_project_root_cache", **{
            key: root_cache[key] for key in ("seed", "actual_epochs", "best_epoch",
            "best_val_loss", "CT", "TRA", "AOGM", "checkpoint_sha256")}},
        {"condition": "same_seed_manual_cache", **{
            key: controlled[0][key] for key in ("seed", "actual_epochs", "best_epoch",
            "best_val_loss", "CT", "TRA", "AOGM")},
         "checkpoint_sha256": prior["checkpoint_sha256"]["replay_seed20839_manual_cache"]},
    ]
    write_csv(out / "cache_reproduction.csv", list(cache_rows[0]), cache_rows)
    write_json(out / "reproducibility_provenance.json", {
        "checkpoint_sha256": prior["checkpoint_sha256"],
        "model_config_sha256": {
            "manual": prior["manual_model_config_sha256"],
            "grid": prior["grid_model_config_sha256"],
            "same_seed_replay": prior["replay_model_config_sha256"],
        },
        "manual_vs_grid_effective_config_differences":
            prior["manual_vs_grid_effective_config_differences"],
        "manual_vs_same_seed_effective_config_differences":
            prior["manual_vs_same_seed_effective_config_differences"],
        "exact_manual_reproduction": prior["exact_manual_reproduction"],
        "runtime": {key: prior["runtime"].get(key) for key in
                    ("torch_version", "lightning_version", "cuda_available", "gpu", "git_commit")},
        "evaluation_set": "02 validation with GT/TRA masks; not independent test",
    })
    write_json(out / "cache_comparison.json", {
        "datasets": [{key: item[key] for key in
                      ("sequence", "cache_key", "images_equal", "gt_masks_equal",
                       "gt_tracks_equal", "crop_random_state_equal")}
                     for item in json_file(val / "cache_comparison.json")["datasets"]]
    })
    shutil.copyfile(val / "seed_stats.json", out / "seed_stats.json")
    for filename in ("ctc_equivalence.csv", "manual_vs_selected_edges.csv"):
        shutil.copyfile(val / filename, out / filename)
    graph_rows = read_csv(val / "graph_summary.csv")
    write_csv(out / "graph_summary.csv",
              [key for key in graph_rows[0] if key != "model_dir"], graph_rows)
    for graph_dir in sorted((val / "graphs").iterdir()):
        if not graph_dir.is_dir():
            continue
        target_graph = out / "validation_graphs" / graph_dir.name
        target_graph.mkdir(parents=True, exist_ok=True)
        for filename in ("false_positive_edges.csv", "false_negative_edges.csv"):
            shutil.copyfile(graph_dir / filename, target_graph / filename)

    mode_rows = read_csv(mode / "comparison.csv")
    mode_columns = [key for key in mode_rows[0] if key != "ctc_result_dir"]
    write_csv(out / "greedy_vs_ilp.csv", mode_columns, mode_rows)
    for filename in ("mode_differences.csv", "changed_edges.csv"):
        shutil.copyfile(mode / filename, out / filename)
    shutil.copyfile(mode / "solver_benchmark.json", out / "solver_benchmark.json")
    mode_provenance = json_file(mode / "provenance.json")
    write_json(out / "greedy_ilp_provenance.json", {
        key: mode_provenance[key] for key in
        ("validation_only", "image_count", "mask_count", "model_sha256",
         "trackastra_version", "motile_version", "ilpy_version",
         "gurobipy_version", "batch_size", "tracking_parameters")
    })
    for name in ("general_2d_greedy", "general_2d_ilp",
                 "finetuned_greedy", "finetuned_ilp"):
        source_graph = mode / name
        target = out / "greedy_ilp_graphs" / name
        target.mkdir(parents=True, exist_ok=True)
        for filename in ("false_positive_edges.csv", "false_negative_edges.csv",
                         "splits.csv", "ctc_metrics.csv"):
            shutil.copyfile(source_graph / filename, target / filename)
    print(f"Curated metadata and graph tables: {out}")


if __name__ == "__main__":
    main()

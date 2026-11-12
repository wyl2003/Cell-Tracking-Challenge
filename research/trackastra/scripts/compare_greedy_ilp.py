"""Compare the installed general_2d and selected fine-tuned model on 02.

No training is performed. Existing model weights and experiment outputs are read only.
All scores are on the 02 validation sequence, not an independent test set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import traceback
import uuid
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

from trackastra.model import Trackastra
from trackastra.data import load_tiff_timeseries
from trackastra.tracking import build_graph, graph_to_ctc, track_greedy
from trackastra.tracking.ilp import track_ilp

from analyze_tracking_graphs import EDGE_COLUMNS, edge_row, gt_edges, sha256, write_csv
from run_finetune_experiments import find_ctc_evaluate, run_command, validate_ctc, write_json


SCRIPTS = Path(__file__).resolve().parent
PROJECT = SCRIPTS.parents[2]
SEQUENCE = PROJECT / "data/Fluo-N2DH-GOWT1/02"
GT = PROJECT / "data/Fluo-N2DH-GOWT1/02_GT"
OUTPUT = PROJECT / "runs/trackastra/greedy_vs_ilp_02"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def metrics(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter=";"))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one metrics row: {path}")
    row = rows[0]
    required = ("TRA", "CT", "AOGM")
    if any(not row.get(key) for key in required):
        raise RuntimeError(f"Missing required metrics in {path}: {row}")
    values = {key: float(row[key]) for key in required}
    for key in ("BC(0)", "BC(1)", "BC(2)", "BC(3)", "gt_divisions",
                "tp_div(0)", "fp_div(0)", "fn_div(0)"):
        try:
            values[key] = float(row[key]) if row.get(key) else None
        except (KeyError, ValueError):
            values[key] = None
    return values


def node_key(attributes: dict) -> tuple[int, int]:
    return (int(attributes["time"]), int(attributes["label"]))


def graph_tables(graph, folder: Path, truth_edges: set) -> dict:
    nodes = {node_key(attrs): attrs for _, attrs in graph.nodes(data=True)}
    edges = {}
    children = defaultdict(set)
    for source, target, attrs in graph.edges(data=True):
        src = node_key(graph.nodes[source])
        dst = node_key(graph.nodes[target])
        if (src, dst) in edges:
            raise RuntimeError(f"Duplicate graph edge {(src, dst)}")
        edges[src, dst] = float(attrs["weight"])
        children[src].add(dst)
    if len(nodes) != graph.number_of_nodes():
        raise RuntimeError("Duplicate (time,label) graph nodes")
    false_positive = set(edges) - truth_edges
    false_negative = truth_edges - set(edges)
    splits = {parent: daughters for parent, daughters in children.items()
              if len(daughters) == 2}
    truth_children = defaultdict(set)
    for source, target in truth_edges:
        truth_children[source].add(target)
    truth_splits = {parent: daughters for parent, daughters in truth_children.items()
                    if len(daughters) == 2}

    write_csv(folder / "nodes.csv", ["time", "label"],
              [{"time": t, "label": label} for t, label in sorted(nodes)])
    write_csv(folder / "edges.csv", EDGE_COLUMNS + ["weight", "in_gt"],
              [dict(edge_row(edge, edges[edge]), in_gt=edge in truth_edges)
               for edge in sorted(edges)])
    write_csv(folder / "false_positive_edges.csv", EDGE_COLUMNS + ["weight"],
              [edge_row(edge, edges[edge]) for edge in sorted(false_positive)])
    write_csv(folder / "false_negative_edges.csv", EDGE_COLUMNS,
              [edge_row(edge) for edge in sorted(false_negative)])
    split_rows = []
    for parent in sorted(set(splits) | set(truth_splits)):
        predicted = splits.get(parent, set())
        expected = truth_splits.get(parent, set())
        split_rows.append({
            "parent_time": parent[0], "parent_label": parent[1],
            "predicted_children": json.dumps(sorted(predicted)),
            "gt_children": json.dumps(sorted(expected)),
            "exact_split": bool(expected and predicted == expected),
            "predicted_split": bool(predicted), "gt_adjacent_split": bool(expected),
        })
    write_csv(folder / "splits.csv",
              ["parent_time", "parent_label", "predicted_children", "gt_children",
               "exact_split", "predicted_split", "gt_adjacent_split"], split_rows)
    return {
        "nodes": len(nodes), "edges": len(edges),
        "false_positive_edges": len(false_positive),
        "false_negative_edges": len(false_negative),
        "predicted_splits": len(splits),
        "gt_adjacent_splits": len(truth_splits),
        "exact_splits": sum(splits.get(parent) == daughters
                            for parent, daughters in truth_splits.items()),
    }


def preflight(models: dict[str, Path], sequence: Path, gt: Path, output: Path,
              ctc_evaluate: str) -> dict:
    for name, folder in models.items():
        for filename in ("model.pt", "config.yaml", "train_config.yaml"):
            if not (folder / filename).is_file():
                raise FileNotFoundError(f"{name}: {folder / filename}")
    frames = sorted(sequence.glob("*.tif"))
    masks = sorted((gt / "TRA").glob("man_track*.tif"))
    if len(frames) != 92 or len(masks) != 92:
        raise RuntimeError(f"Expected 92 images and 92 GT/TRA masks; got {len(frames)}, {len(masks)}")
    if not (gt / "TRA/man_track.txt").is_file():
        raise FileNotFoundError(gt / "TRA/man_track.txt")
    if not Path(ctc_evaluate).is_file():
        raise FileNotFoundError(ctc_evaluate)
    return {
        "sequence": str(sequence), "gt": str(gt), "output": str(output),
        "validation_only": True, "image_count": len(frames), "mask_count": len(masks),
        "model_dirs": {name: str(folder) for name, folder in models.items()},
        "model_sha256": {name: sha256(folder / "model.pt") for name, folder in models.items()},
        "trackastra_version": importlib.metadata.version("trackastra"),
        "motile_version": importlib.metadata.version("motile"),
        "ilpy_version": importlib.metadata.version("ilpy"),
        "gurobipy_version": importlib.metadata.version("gurobipy"),
        "ctc_evaluate": ctc_evaluate, "batch_size": 1,
        "tracking_parameters": {"greedy_threshold": 0.5, "ilp_config": "gt",
                                "ilp_allow_divisions": True,
                                "ilp_solver": "motile.Solver.solve defaults: backend Any, 1 thread, no timeout"},
    }


def run_group(name: str, mode: str, model_dir: Path, sequence: Path, gt: Path,
              root: Path, device: str, evaluator: str, truth_edges: set) -> dict:
    folder = root / f"{name}_{mode}"
    folder.mkdir(parents=True, exist_ok=True)
    result_file = folder / "result.json"
    expected_sha = sha256(model_dir / "model.pt")
    if result_file.is_file():
        existing = json.loads(result_file.read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and existing.get("model_sha256") == expected_sha:
            print(f"SKIP {name} {mode}: complete", flush=True)
            return existing
        raise RuntimeError(f"Existing result needs inspection; refusing overwrite: {result_file}")
    final_res = folder / "02_RES"
    if final_res.exists():
        raise FileExistsError(f"Refusing to overwrite CTC result: {final_res}")
    stage = folder / f"02_RES.partial_{uuid.uuid4().hex[:8]}"
    result = {"model": name, "mode": mode, "model_dir": str(model_dir),
              "model_sha256": expected_sha, "status": "running",
              "sequence": str(sequence), "masks": str(gt / "TRA"),
              "ctc_result_dir": str(final_res), "evaluation_set": "02 validation (not independent test)"}
    write_json(result_file, result)
    try:
        start = time.perf_counter()
        model = Trackastra.from_folder(model_dir, device=device, batch_size=1)
        load_done = time.perf_counter()
        print(f"TRACK {name} {mode}", flush=True)
        graph, masks = model.track_from_disk(sequence, gt / "TRA", mode=mode)
        track_done = time.perf_counter()
        graph_stats = graph_tables(graph, folder, truth_edges)
        graph_done = time.perf_counter()
        graph_to_ctc(graph, masks, outdir=stage)
        (stage / "man_track.txt").rename(stage / "res_track.txt")
        for path in stage.glob("man_track*.tif"):
            path.rename(stage / path.name.replace("man_track", "mask", 1))
        validate_ctc(stage, gt)
        stage.rename(final_res)
        export_done = time.perf_counter()
        metric_file = folder / "ctc_metrics.csv"
        run_command([evaluator, "--res", str(final_res), "--gt", str(gt),
                     "--tra", "--ct", "--bc", "1", "-n", "1",
                     "--csv-file", str(metric_file)], folder / "metrics.log")
        metric_done = time.perf_counter()
        result.update(graph_stats)
        result.update(metrics(metric_file))
        result.update({
            "status": "complete", "model_load_s": load_done - start,
            "tracking_s": track_done - load_done,
            "graph_export_s": graph_done - track_done,
            "ctc_export_s": export_done - graph_done,
            "metrics_s": metric_done - export_done,
            "total_s": metric_done - start,
        })
        write_json(result_file, result)
        print(f"DONE {name} {mode}: CT={result['CT']:.6f}, TRA={result['TRA']:.6f}", flush=True)
        return result
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        write_json(result_file, result)
        (folder / "errors.log").write_text(result["error"], encoding="utf-8")
        raise


def summarize(root: Path, truth_edges: set, skipped_parent_links: int) -> None:
    names = ("general_2d_greedy", "general_2d_ilp", "finetuned_greedy", "finetuned_ilp")
    records = [json.loads((root / name / "result.json").read_text(encoding="utf-8"))
               for name in names]
    if any(row["status"] != "complete" for row in records):
        raise RuntimeError("Not all four experiments are complete")
    columns = ["model", "mode", "TRA", "CT", "AOGM", "BC(0)", "BC(1)",
               "nodes", "edges", "false_positive_edges", "false_negative_edges",
               "predicted_splits", "gt_adjacent_splits", "exact_splits",
               "model_load_s", "tracking_s", "ctc_export_s", "metrics_s", "total_s",
               "model_sha256", "ctc_result_dir", "evaluation_set"]
    write_csv(root / "comparison.csv", columns,
              [{column: record.get(column, "") for column in columns}
               for record in records])
    sets = {}
    for name in names:
        folder = root / name
        nodes = {(int(row["time"]), int(row["label"]))
                 for row in read_csv(folder / "nodes.csv")}
        edges = {((int(row["source_time"]), int(row["source_label"])),
                  (int(row["target_time"]), int(row["target_label"])))
                 for row in read_csv(folder / "edges.csv")}
        splits = {(int(row["parent_time"]), int(row["parent_label"]))
                  for row in read_csv(folder / "splits.csv") if row["predicted_split"] == "True"}
        sets[name] = (nodes, edges, splits)
    pair_rows = []
    for model in ("general_2d", "finetuned"):
        greedy = sets[f"{model}_greedy"]
        ilp = sets[f"{model}_ilp"]
        pair_rows.append({
            "model": model,
            "nodes_only_greedy": len(greedy[0] - ilp[0]),
            "nodes_only_ilp": len(ilp[0] - greedy[0]),
            "edges_only_greedy": len(greedy[1] - ilp[1]),
            "edges_only_ilp": len(ilp[1] - greedy[1]),
            "splits_only_greedy": len(greedy[2] - ilp[2]),
            "splits_only_ilp": len(ilp[2] - greedy[2]),
            "exact_same_graph": greedy[0] == ilp[0] and greedy[1] == ilp[1],
        })
    write_csv(root / "mode_differences.csv", list(pair_rows[0]), pair_rows)
    edge_rows = []
    for edge in sorted(set.union(*(value[1] for value in sets.values()))):
        present = {name: edge in sets[name][1] for name in names}
        if len(set(present.values())) == 1:
            continue
        edge_rows.append({**edge_row(edge), "in_gt": edge in truth_edges, **present})
    write_csv(root / "changed_edges.csv", EDGE_COLUMNS + ["in_gt"] + list(names), edge_rows)
    write_json(root / "summary.json", {
        "evaluation_set": "02 validation (not independent test)",
        "gt_adjacent_edge_count": len(truth_edges),
        "gt_parent_links_with_frame_gaps_excluded": skipped_parent_links,
        "mode_differences": pair_rows, "changed_edge_count": len(edge_rows),
        "results": records,
    })


def benchmark_solvers(models: dict[str, Path], sequence: Path, gt: Path,
                      root: Path, device: str) -> None:
    """Time both algorithms on one identical candidate graph per model."""
    output = root / "solver_benchmark.json"
    if output.exists():
        print(f"SKIP solver benchmark: {output}", flush=True)
        return
    imgs = load_tiff_timeseries(sequence)
    masks = load_tiff_timeseries(gt / "TRA")
    results = {}
    for name, folder in models.items():
        model = Trackastra.from_folder(folder, device=device, batch_size=1)
        start = time.perf_counter()
        predictions = model._predict(imgs, masks, batch_size=1)
        predicted = time.perf_counter()
        candidate = build_graph(nodes=predictions["nodes"],
                                weights=predictions["weights"],
                                use_distance=False, max_distance=256,
                                max_neighbors=10, delta_t=1)
        built = time.perf_counter()
        record = {
            "prediction_s": predicted - start,
            "candidate_build_s": built - predicted,
            "candidate_nodes": candidate.number_of_nodes(),
            "candidate_edges": candidate.number_of_edges(),
        }
        for mode, algorithm in (("greedy", track_greedy), ("ilp", track_ilp)):
            start_solver = time.perf_counter()
            graph = algorithm(candidate) if mode == "greedy" else algorithm(candidate, ilp_config="gt")
            finished = time.perf_counter()
            generated_nodes = {node_key(attrs) for _, attrs in graph.nodes(data=True)}
            generated_edges = {(node_key(graph.nodes[u]), node_key(graph.nodes[v]))
                               for u, v in graph.edges}
            saved = root / f"{name}_{mode}"
            saved_nodes = {(int(row["time"]), int(row["label"]))
                           for row in read_csv(saved / "nodes.csv")}
            saved_edges = {((int(row["source_time"]), int(row["source_label"])),
                            (int(row["target_time"]), int(row["target_label"])))
                           for row in read_csv(saved / "edges.csv")}
            record[mode] = {
                "solver_s": finished - start_solver,
                "nodes": len(generated_nodes), "edges": len(generated_edges),
                "same_nodes_as_main_run": generated_nodes == saved_nodes,
                "same_edges_as_main_run": generated_edges == saved_edges,
            }
        results[name] = record
    write_json(output, {"method": "one prediction and one candidate graph per model; "
                         "default build_graph parameters; one timed call per solver",
                        "results": results})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", type=Path, required=True,
                        help="Installed general_2d model folder")
    parser.add_argument("--finetuned", type=Path, required=True,
                        help="Selected fine-tuned folder containing the best model.pt")
    parser.add_argument("--sequence", type=Path, default=SEQUENCE)
    parser.add_argument("--gt", type=Path, default=GT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--ctc-evaluate", type=str)
    parser.add_argument("--benchmark-solvers", action="store_true",
                        help="Run a same-candidate graph solver timing check")
    args = parser.parse_args()
    models = {"general_2d": args.pretrained.resolve(), "finetuned": args.finetuned.resolve()}
    sequence, gt, root = args.sequence.resolve(), args.gt.resolve(), args.output_root.resolve()
    evaluator = find_ctc_evaluate(args.ctc_evaluate)
    provenance = preflight(models, sequence, gt, root, evaluator)
    root.mkdir(parents=True, exist_ok=True)
    provenance_file = root / "provenance.json"
    if provenance_file.exists():
        existing = json.loads(provenance_file.read_text(encoding="utf-8"))
        if existing != provenance:
            raise RuntimeError("Existing output provenance differs; use a new output-root")
    else:
        write_json(provenance_file, provenance)
    truth_edges, skipped = gt_edges(gt / "TRA")
    write_csv(root / "gt_adjacent_edges.csv", EDGE_COLUMNS,
              [edge_row(edge) for edge in sorted(truth_edges)])
    for name, folder in models.items():
        for mode in ("greedy", "ilp"):
            run_group(name, mode, folder, sequence, gt, root, args.device, evaluator, truth_edges)
    summarize(root, truth_edges, skipped)
    if args.benchmark_solvers:
        benchmark_solvers(models, sequence, gt, root, args.device)
    print(f"SUMMARY {root / 'comparison.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

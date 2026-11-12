"""Compare the six metric-tied models with each other and manual Exp 2.

Inference uses the same 02 images, GT/TRA masks, greedy mode and batch size 1
as experiments/compare_models.py. Outputs stay under finetune_validation_02.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import tifffile
import yaml

from trackastra.model import Trackastra

from run_finetune_experiments import (
    SCRIPTS_DIR,
    best_validation,
    find_ctc_evaluate,
    read_metrics,
    run_command,
    write_json,
)


Edge = tuple[tuple[int, int], tuple[int, int]]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gt_edges(tra_dir: Path) -> tuple[set[Edge], int]:
    tracks = {}
    for line in (tra_dir / "man_track.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            track_id, start, end, parent = map(int, line.split())
            tracks[track_id] = (start, end, parent)
    edges = set()
    skipped_parent_links = 0
    for track_id, (start, end, parent) in tracks.items():
        for frame in range(start, end):
            edges.add(((frame, track_id), (frame + 1, track_id)))
        if parent:
            parent_end = tracks[parent][1]
            if parent_end + 1 == start:
                edges.add(((parent_end, parent), (start, track_id)))
            else:
                skipped_parent_links += 1
    return edges, skipped_parent_links


def edge_row(edge: Edge, weight: float | None = None) -> dict:
    (t1, label1), (t2, label2) = edge
    row = {"source_time": t1, "source_label": label1,
           "target_time": t2, "target_label": label2}
    if weight is not None:
        row["weight"] = weight
    return row


EDGE_COLUMNS = ["source_time", "source_label", "target_time", "target_label"]


def ctc_equivalence(reference: Path, candidate: Path) -> dict:
    """Check tracking equivalence after one consistent CTC track-ID relabeling."""
    ref_masks = sorted(reference.glob("mask*.tif"))
    candidate_masks = sorted(candidate.glob("mask*.tif"))
    if len(ref_masks) != len(candidate_masks) or not ref_masks:
        raise ValueError("CTC mask frame counts differ")
    mapping = {0: 0}
    raw_pixel_equal_frames = 0
    same_foreground = True
    consistent_mapping = True
    for ref_path, candidate_path in zip(ref_masks, candidate_masks):
        ref = tifffile.imread(ref_path)
        other = tifffile.imread(candidate_path)
        if ref.shape != other.shape:
            raise ValueError("CTC mask shapes differ")
        raw_pixel_equal_frames += bool(np.array_equal(ref, other))
        same_foreground &= bool(np.array_equal(ref > 0, other > 0))
        stride = int(ref.max()) + 1
        pairs = np.unique(other.astype(np.uint64) * stride + ref.astype(np.uint64))
        for pair in pairs:
            candidate_id, reference_id = divmod(int(pair), stride)
            if candidate_id == 0 or reference_id == 0:
                continue
            previous = mapping.setdefault(candidate_id, reference_id)
            if previous != reference_id:
                consistent_mapping = False
    bijective = len(set(mapping.values())) == len(mapping)
    def track_rows(path: Path) -> list[tuple[int, int, int, int]]:
        rows = []
        for line in (path / "res_track.txt").read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(tuple(map(int, line.split())))
        return rows
    ref_tracks = sorted(track_rows(reference))
    candidate_tracks = track_rows(candidate)
    try:
        relabeled_tracks = sorted((mapping[track], start, end, mapping[parent])
                                  for track, start, end, parent in candidate_tracks)
    except KeyError:
        relabeled_tracks = []
    track_table_equal = ref_tracks == relabeled_tracks
    return {
        "same_foreground": same_foreground,
        "consistent_track_id_mapping": consistent_mapping,
        "bijective_track_id_mapping": bijective,
        "track_table_equal_after_relabel": track_table_equal,
        "semantically_equivalent": same_foreground and consistent_mapping and bijective and track_table_equal,
        "raw_mask_pixel_equal_frames": raw_pixel_equal_frames,
        "total_frames": len(ref_masks),
        "track_id_count": len(mapping) - 1,
    }


def load_or_infer(name: str, model_dir: Path, sequence: Path,
                  output_root: Path, device: str) -> tuple[set[tuple[int, int]], dict[Edge, float]]:
    folder = output_root / "graphs" / name
    meta_path = folder / "metadata.json"
    nodes_path = folder / "nodes.csv"
    edges_path = folder / "edges.csv"
    weight_sha = sha256(model_dir / "model.pt")
    if meta_path.is_file() and nodes_path.is_file() and edges_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("model_sha256") == weight_sha:
            nodes = {(int(row["time"]), int(row["label"]))
                     for row in read_csv(nodes_path)}
            edges = {
                ((int(row["source_time"]), int(row["source_label"])),
                 (int(row["target_time"]), int(row["target_label"]))): float(row["weight"])
                for row in read_csv(edges_path)
            }
            print(f"CACHED GRAPH {name}", flush=True)
            return nodes, edges

    print(f"INFER GRAPH {name}", flush=True)
    model = Trackastra.from_folder(model_dir, device=device, batch_size=1)
    graph, _ = model.track_from_disk(sequence, sequence.parent / "02_GT" / "TRA",
                                     mode="greedy")
    nodes = {
        (int(attributes["time"]), int(attributes["label"]))
        for _, attributes in graph.nodes(data=True)
    }
    edges = {}
    for source, target, attributes in graph.edges(data=True):
        source_node = graph.nodes[source]
        target_node = graph.nodes[target]
        key = ((int(source_node["time"]), int(source_node["label"])),
               (int(target_node["time"]), int(target_node["label"])))
        if key in edges:
            raise RuntimeError(f"Repeated edge key in {name}: {key}")
        edges[key] = float(attributes["weight"])
    folder.mkdir(parents=True, exist_ok=True)
    write_csv(nodes_path, ["time", "label"],
              [{"time": time, "label": label} for time, label in sorted(nodes)])
    write_csv(edges_path, EDGE_COLUMNS + ["weight"],
              [edge_row(edge, edges[edge]) for edge in sorted(edges)])
    write_json(meta_path, {"model_dir": str(model_dir), "model_sha256": weight_sha,
                           "node_count": len(nodes), "edge_count": len(edges),
                           "sequence": str(sequence), "masks": str(sequence.parent / "02_GT" / "TRA"),
                           "mode": "greedy", "batch_size": 1})
    return nodes, edges


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    project = SCRIPTS_DIR.parents[2]
    parser.add_argument("--sweep-root", type=Path,
                        default=project / "runs/trackastra/finetune_sweep_01")
    parser.add_argument("--seed-root", type=Path,
                        default=project / "runs/trackastra/finetune_seed_repeats")
    parser.add_argument("--manual-model-dir", type=Path, required=True)
    parser.add_argument("--manual-res", type=Path, required=True)
    parser.add_argument("--output-root", type=Path,
                        default=project / "runs/trackastra/finetune_validation_02")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--ctc-evaluate", type=str)
    args = parser.parse_args(argv)
    sweep = read_csv(args.sweep_root / "results.csv")
    if len(sweep) != 8 or any(row["status"] != "complete" for row in sweep):
        raise ValueError("Expected all eight completed grid-search runs")
    best_ct = max(float(row["CT"]) for row in sweep)
    best_tra = max(float(row["TRA"]) for row in sweep)
    tied = [row for row in sweep if math.isclose(float(row["CT"]), best_ct, abs_tol=1e-12)
            and math.isclose(float(row["TRA"]), best_tra, abs_tol=1e-12)]
    if len(tied) != 6:
        raise ValueError(f"Expected six tied models, found {len(tied)}")
    chosen = min(tied, key=lambda row: float(row["best_val_loss"]))
    config = yaml.safe_load(Path(chosen["config_file"]).read_text(encoding="utf-8"))
    sequence = Path(config["input_val"][0]).resolve()
    gt = sequence.parent / "02_GT"
    truth, skipped_parent_links = gt_edges(gt / "TRA")
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "gt_edges.csv", EDGE_COLUMNS,
              [edge_row(edge) for edge in sorted(truth)])

    manual_metrics_file = args.output_root / "manual_exp2_metrics.csv"
    if not manual_metrics_file.is_file():
        run_command([find_ctc_evaluate(args.ctc_evaluate), "--res", str(args.manual_res),
                     "--gt", str(gt), "--tra", "--ct", "-n", "1",
                     "--csv-file", str(manual_metrics_file)],
                    args.output_root / "manual_exp2_metrics.log")
    manual_metrics = read_metrics(manual_metrics_file)
    manual_epochs, manual_best_epoch, manual_loss = best_validation(args.manual_model_dir)
    write_json(args.output_root / "manual_exp2.json", {
        "model_dir": str(args.manual_model_dir.resolve()),
        "ctc_result_dir": str(args.manual_res.resolve()),
        "actual_epochs": manual_epochs, "best_epoch": manual_best_epoch,
        "best_val_loss": manual_loss, **manual_metrics,
    })

    models = [(row["experiment"], Path(row["best_model_dir"]), row) for row in tied]
    seed_runs = read_csv(args.seed_root / "results.csv")
    if len(seed_runs) != 3 or any(row["status"] != "complete" for row in seed_runs):
        raise ValueError("Expected three completed seed-repeat runs")
    models.extend((f"seed_{row['seed']}", Path(row["best_model_dir"]), row)
                  for row in seed_runs)
    models.append(("manual_exp2", args.manual_model_dir, {"seed": "20839", **manual_metrics}))
    all_graphs = {}
    for name, model_dir, _ in models:
        all_graphs[name] = load_or_infer(name, model_dir, sequence,
                                        args.output_root, args.device)
    reference_nodes, reference_weights = all_graphs[chosen["experiment"]]
    reference_edges = set(reference_weights)
    summaries = []
    for name, model_dir, source in models:
        nodes, weights = all_graphs[name]
        edges = set(weights)
        fp, fn = edges - truth, truth - edges
        folder = args.output_root / "graphs" / name
        write_csv(folder / "false_positive_edges.csv", EDGE_COLUMNS + ["weight"],
                  [edge_row(edge, weights[edge]) for edge in sorted(fp)])
        write_csv(folder / "false_negative_edges.csv", EDGE_COLUMNS,
                  [edge_row(edge) for edge in sorted(fn)])
        common = reference_edges & edges
        weight_deltas = [abs(weights[edge] - reference_weights[edge]) for edge in common]
        summaries.append({
            "model": name, "model_dir": str(model_dir), "seed": source.get("seed", ""),
            "CT": source.get("CT", ""), "TRA": source.get("TRA", ""),
            "nodes": len(nodes), "edges": len(edges), "FP_edges": len(fp),
            "FN_edges": len(fn), "same_nodes_as_chosen": nodes == reference_nodes,
            "same_edges_as_chosen": edges == reference_edges,
            "different_edge_weights": sum(delta > 1e-6 for delta in weight_deltas),
            "max_abs_edge_weight_delta": max(weight_deltas, default=0.0),
        })
    write_csv(args.output_root / "graph_summary.csv", list(summaries[0]), summaries)

    ctc_rows = []
    for name, _, source in models:
        ctc_dir = args.manual_res if name == "manual_exp2" else Path(source["ctc_result_dir"])
        ctc_rows.append({"model": name, **ctc_equivalence(
            Path(chosen["ctc_result_dir"]), ctc_dir)})
    write_csv(args.output_root / "ctc_equivalence.csv", list(ctc_rows[0]), ctc_rows)

    manual_nodes, manual_weights = all_graphs["manual_exp2"]
    manual_edges = set(manual_weights)
    difference_rows = []
    for edge in sorted(reference_edges ^ manual_edges):
        difference_rows.append({
            **edge_row(edge),
            "status": "selected_only" if edge in reference_edges else "manual_only",
            "in_gt": edge in truth,
            "selected_weight": reference_weights.get(edge, ""),
            "manual_weight": manual_weights.get(edge, ""),
        })
    write_csv(args.output_root / "manual_vs_selected_edges.csv",
              EDGE_COLUMNS + ["status", "in_gt", "selected_weight", "manual_weight"],
              difference_rows)
    six_nodes = [all_graphs[row["experiment"]][0] for row in tied]
    six_edges = [set(all_graphs[row["experiment"]][1]) for row in tied]
    write_json(args.output_root / "comparison.json", {
        "sequence": str(sequence), "gt_tra": str(gt / "TRA"),
        "gt_edge_count": len(truth), "skipped_parent_links": skipped_parent_links,
        "metric_tie_count": len(tied), "all_six_nodes_identical": all(x == six_nodes[0] for x in six_nodes),
        "all_six_edges_identical": all(x == six_edges[0] for x in six_edges),
        "chosen_for_comparison": chosen["experiment"],
        "manual_vs_chosen_added_edges": len(reference_edges - manual_edges),
        "manual_vs_chosen_removed_edges": len(manual_edges - reference_edges),
        "manual_vs_chosen_node_difference": len(reference_nodes ^ manual_nodes),
    })
    print(f"Graph analysis saved to {args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

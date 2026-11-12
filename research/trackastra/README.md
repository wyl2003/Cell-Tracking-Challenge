# Trackastra research: fine-tuning, tracking decoders and Multi-scale segmentation

This directory records completed Windows experiments on Fluo-N2DH-GOWT1. It is separate from the team's `src/` baseline and Cellpose modules. It contains no pretrained or fine-tuned weights, CTC images, datasets, caches, or virtual environments.

**All scores here are on the Fluo-N2DH-GOWT1 `02` validation sequence.** The association experiments feed GT/TRA masks to the tracker. The end-to-end experiments feed predicted Cellpose masks to the tracker and use GT for scoring.

The integrated report contains the original TensorBoard epoch loss curves, comparison tables, and conclusions. [Integrated report](reports/integrated_conclusion.md)

## Results

- [Fine-tuning and eight-run grid](reports/grid_search.md): six CT/TRA ties; the selected checkpoint is `e10_s89_lr1e-05_172ea7de19d3`, chosen by Validation Loss among the tied runs.
- [Three-seed stability and cache reproducibility](reports/reproducibility.md): seeds 43, 44, 20839; CT `0.893728 ± 0.006007` and TRA `0.996612 ± 0.000020` (sample SD, n=3). The manual seed-20839 run was reproduced exactly when its cache was reused; the root cache did not reproduce it.
- With GT/TRA masks, `general_2d + ILP` has the highest CT (`0.964912`) and TRA (`0.999359`). ILP keeps eight detections omitted by Greedy and selects more wrong edges. [Greedy versus ILP](reports/greedy_vs_ilp.md)
- With predicted masks, Multi-scale improves TRA and CT in all four paired comparisons. The pretrained ILP run has the highest TRA (`0.96198`) and OP_CTB (`0.94060`). The pretrained Greedy run has the highest CT (`0.37288`). [End-to-end comparison](results/multiscale_tracking_02/comparison.csv)
- [Four-row comparison CSV](results/greedy_vs_ilp.csv), [changed edges](results/changed_edges.csv), [solver timing](results/solver_benchmark.json), and [curated result directory](results/).

## Layout

| Path | Purpose |
|---|---|
| `configs/finetune_config.example.yaml` | Local-path template retaining the original model, training and validation settings |
| `scripts/run_finetune_experiments.py` | Grid and seed runner; starts every attempt from `general_2d`, records best checkpoint and CTC metrics |
| `scripts/analyze_tracking_graphs.py` | Compare six tied graphs, three seed reruns and the manual run |
| `scripts/compare_greedy_ilp.py` | Four model × decoder runs, independent CTC directories, BC/CT/TRA/AOGM and graph errors |
| `scripts/run_multiscale_tracking.py` | Generate single-scale and Multi-scale Cellpose masks, then evaluate all eight model × decoder × mask combinations |
| `scripts/curate_results.py` | Export path-free, small CSV/JSON/graph artifacts from local completed runs |
| `results/` | Numerical results, checkpoint hashes, changed/error edge tables and raw metric CSVs; no full GT graph or TIFF masks |
| `reports/` | Integrated findings, individual experiment reports and the epoch loss figure |

## Reproduce the workflow

Use a separate Python environment with the **official Trackastra 0.5.6 checkout** (`aa57a95` for the recorded run), its training dependencies, TensorBoard, PyYAML and py-ctcmetrics 1.3.3. ILP needs Motile and a working ilpy solver backend; the recorded environment used Motile 1.0.1, ilpy 0.6.0, and had gurobipy 13.0.3 installed. Check the solver license/backend locally before rerunning. The target repository's `requirements.txt` remains dedicated to the existing scaffold.

Copy `configs/finetune_config.example.yaml` to a local, untracked YAML and set the three paths (`model`, `input_train`, `input_val`). Download CTC data separately and keep it under ignored `data/`. The original checkpoint is **not** in this repository; place `general_2d` in an external model directory. From this repository root, Windows PowerShell examples are:

```powershell
$py = "C:\path\to\trackastra-env\python.exe"
$official = "C:\path\to\official-trackastra"
$config = "C:\path\to\local-finetune-config.yaml"

# Review the original eight combinations without training.
& $py research/trackastra/scripts/run_finetune_experiments.py --config $config --trackastra-root $official --output-root runs/trackastra/finetune_sweep_01 --epochs 2 10 --train-samples 16 89 --lrs 1e-5 5e-6 --seed 42 --dry-run

# Only if intentionally reproducing the historical training: remove --dry-run.
# Then run three fixed-parameter seed repetitions into a separate output root.
& $py research/trackastra/scripts/run_finetune_experiments.py --config $config --trackastra-root $official --output-root runs/trackastra/finetune_seed_repeats --epochs 10 --train-samples 89 --lrs 1e-5 --seed 43

# Evaluate the already selected best model under both decoders; no training.
& $py research/trackastra/scripts/compare_greedy_ilp.py --pretrained C:\path\to\general_2d --finetuned C:\path\to\selected\train --sequence data/Fluo-N2DH-GOWT1/02 --gt data/Fluo-N2DH-GOWT1/02_GT --output-root runs/trackastra/greedy_vs_ilp_02 --device cuda --benchmark-solvers
```

The runner's `--cache-dir` pins the official training cache (default `<trackastra-root>/.cache`). Cache state affects crop randomness. Repeating a seed with a different cache does not guarantee the same weights. Training uses official `scripts/train.py`; best-epoch `model.pt` is used for inference, while `last.ckpt` is only a resume checkpoint. `ctc_evaluate` must be on PATH or supplied with `--ctc-evaluate <absolute executable path>`. New runs are written under ignored `runs/`; the fine-tuning runner skips completed runs with a valid checkpoint and CTC result, while the decoder comparison also checks checkpoint hashes before skipping. Both keep unrelated CTC outputs intact.

To regenerate the small committed tables from a local copy of the completed experiments, use `scripts/curate_results.py --local-scripts <official-checkout>/scripts`. The local run directory must contain the original output folders; this command only reads them and writes metadata. The committed reports and numeric results remain useful when the large private artifacts are unavailable.

## Reproduce the Multi-scale Cellpose comparison

Use a Trackastra environment with Cellpose `4.2.1.1` and the `cpsam_v2` weights available through the Cellpose model cache. The runner applies the team's merge function from commit `72ed2c1` using diameters 48 and 32, overlap threshold 0.10, and minimum area 200 pixels. The checkpoint and source hashes are recorded with the results. [Run settings](results/multiscale_tracking_02/provenance.json)

Run these commands from the repository root. Create the detached worktree once, then set the local checkpoint and evaluator paths. Both model folders should contain `model.pt`, `config.yaml`, and `train_config.yaml`.

```powershell
git fetch https://github.com/wyl2003/Cell-Tracking-Challenge.git feature/segmentation-improvement
git worktree add --detach runs/trackastra/segmentation_source 72ed2c19d55ac606cf28f3d928e97e8362301248

& $py research/trackastra/scripts/run_multiscale_tracking.py `
  --dataset data/Fluo-N2DH-GOWT1 --sequence 02 `
  --segmentation-source runs/trackastra/segmentation_source/src/segmentation_postprocess.py `
  --pretrained C:\path\to\general_2d --finetuned C:\path\to\selected\train `
  --output runs/trackastra/multiscale_tracking_02 --device cuda `
  --ctc-evaluate C:\path\to\ctc_evaluate.exe
```

The runner writes the per-frame instance masks and one CTC result directory per configuration under the output folder. The archived scores and per-frame instance counts can be inspected directly. [Results directory](results/multiscale_tracking_02/)

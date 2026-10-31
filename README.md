# COMP9444 Cell Tracking scaffold

Minimal, tested starting point for the Cell Tracking Challenge project
(COMP9444 26T3, Project 15). It does **not** contain a segmentation network —
that is the part you build. It contains everything around it that is fiddly and
easy to get wrong: CTC file conventions, a tracking-by-detection baseline, and
the official metric wiring.

```
src/ctc_io.py             read/write man_track.txt, res_track.txt, mask###.tif; consistency checks
src/track.py              IoU + Hungarian association, gap closing, mitosis (1 -> 2) detection
src/run_track.py          per-frame masks  ->  <seq>_RES/ (mask###.tif + res_track.txt)
src/run_trackastra.py     same masks, linked by Trackastra instead (the SOTA-linker reference row)
src/evaluate.py           <seq>_GT + <seq>_RES -> DET / SEG / TRA / OP_CTB via py-ctcmetrics
src/diagnostics.py        the six AOGM error terms behind TRA, plus CT / TF / division recall
src/error_analysis.py     what the FNs and FPs actually are (brightness, area, border, overlap)
src/make_figure.py        report figures: segmentation panels and the baseline comparison bars
src/segment_cellpose.py   zero-shot Cellpose (cpsam or nuclei) -> mask###.tif (the baseline)
src/segment_classical.py  Otsu + distance-transform watershed -> mask###.tif (the floor)
tests/test_pipeline.py    synthetic mother->two-daughters sequence, both 0- and 1-based naming
```

## Install

Everything goes in a project-local virtual environment; nothing is installed
into the system Python.

```bash
python -m venv .venv --system-site-packages   # reuses the existing torch/numpy wheels
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # macOS / Linux
.venv/Scripts/python tests/test_pipeline.py   # should print SEG=DET=TRA=1.0000 for both cases
```

All commands below assume `.venv/Scripts/python` (Windows) or `.venv/bin/python`.

GPU: the system torch is a CPU build. The venv carries `torch 2.14.1+cu126`, which
shadows it, so CUDA works inside the venv only (the system Python is untouched):

```bash
.venv/Scripts/python -m pip install --index-url https://download.pytorch.org/whl/cu126 "torch==2.14.1+cu126"
```

## Use

```bash
# 1. segment (zero-shot baseline; swap in your own model here)
.venv/Scripts/python -m src.segment_cellpose --seq data/Fluo-N2DH-GOWT1/01 --out runs/cpsam/01 --model cpsam

# 2. track + write a CTC result folder
.venv/Scripts/python -m src.run_track --masks runs/cpsam/01 --out runs/01_RES --iou 0.2

# 3. score it with the official metrics
.venv/Scripts/python -m src.evaluate --gt data/Fluo-N2DH-GOWT1/01_GT --res runs/01_RES

# 4. and see *what* the remaining error is made of
.venv/Scripts/python -m src.diagnostics --gt data/Fluo-N2DH-GOWT1/01_GT --res runs/01_RES
.venv/Scripts/python -m src.error_analysis --seq data/Fluo-N2DH-GOWT1/01 --res runs/01_RES --iou 0.0
```

Oracle ablation (upper bound for tracking, isolates segmentation error):

```bash
.venv/Scripts/python -m src.run_track --from-gt data/Fluo-N2DH-GOWT1/01_GT --out runs/oracle_01_RES
.venv/Scripts/python -m src.evaluate --gt data/Fluo-N2DH-GOWT1/01_GT --res runs/oracle_01_RES
```

SOTA linker reference row (same masks, only the linker changes):

```bash
.venv/Scripts/python -m pip install trackastra
.venv/Scripts/python -m src.run_trackastra --img data/Fluo-N2DH-GOWT1/01 --masks runs/cpsam/01 \
    --out runs/01_RES_astra_g2d --model general_2d --mode greedy
```

Trackastra (`general_2d`, greedy) ties our IoU+Hungarian on the headline metric —
AOGM 422 vs 424 on 01, 1245 vs 1230 on 02, TRA identical at 0.9821 on 01 — while
being clearly better on track quality (02: 58 tracks vs 76, true value 58; TF
0.930 vs 0.914; spurious divisions 1 vs 7). Neither linker gets a single division
right, including the CTC-trained `ctc` weights. Full tables, the oracle-mask
control and the untested `ilp` mode: [plan/RESULTS_linker_comparison.md](plan/RESULTS_linker_comparison.md).

Measured on Fluo-N2DH-GOWT1, both sequences, same tracker, all zero-shot:

| setting | DET 01 / 02 | SEG 01 / 02 | TRA 01 / 02 |
|---|---|---|---|
| classical Otsu + watershed | — | recall ceiling 0.49 | — |
| Cellpose 3 `nuclei`, defaults | 0.8573 / 0.7563 | 0.6662 / 0.6827 | 0.8565 / 0.7550 |
| Cellpose 3 `nuclei`, `cellprob=3, flow=1` | 0.8214 / 0.6599 | 0.3589 / 0.4225 | 0.8195 / 0.6562 |
| **Cellpose 4 `cpsam`, defaults** | **0.9818 / 0.9570** | **0.8878 / 0.9192** | **0.9821 / 0.9574** |

`cpsam` is the model to use. It costs 2.18 s/frame and 1.84 GB of VRAM on an
RTX 3060 Laptop (~3.5 min per 92-frame sequence), which is cheap enough that the
whole ablation grid is practical. Full breakdown, including the six AOGM error
terms and why TRA 0.98 still hides a 0/6 division recall, is in
[plan/RESULTS_cpsam.md](plan/RESULTS_cpsam.md).

Reproducibility: the same model run on GPU vs CPU gives identical SEG on seq 01
(0.6662) but slightly different DET/TRA (0.8573/0.8565 vs 0.8531/0.8518) and 47 vs
54 tracks, because a handful of masks differ by a pixel or two and the
association then jitters. Fix the device in a comparison table and say so in the
report.

The third row is a deliberate negative result: tuning looked better on a 6-frame
per-cell-IoU proxy and was much worse under the official metric, which is
recall-dominated. Tune and validate on full sequences with `src.evaluate`.

Wall clock for Cellpose `nuclei` at 1024x1024: **0.46 s/frame on an RTX 3060 Laptop**
(GPU) vs 2.0-2.6 s/frame on a 20-core CPU, so a full 92-frame sequence is ~45 s
with CUDA. `src/segment_cellpose.py` uses the GPU automatically when torch reports
CUDA and `--cpu` forces the fallback.

Cellpose 4 ships only the large `cpsam` / `cpdino` ViT models (1.15 GB of weights,
no `nuclei`/`cyto3` equivalent). `.venv` therefore carries **4.2.1.1**; to
reproduce the `nuclei` rows you have to `pip install cellpose==3.1.1.2` first.
The two versions cannot coexist in one environment.

## Notes

* Results are written 0-based (`mask000.tif` = first frame). `src/evaluate.py`
  renumbers a temporary copy of both GT and results, so sequences numbered from
  1 are scored correctly too — this is a real off-by-one trap in the metric code.
* CTC data may not be redistributed or committed to a repository, and the
  download host `data.celltrackingchallenge.net` is unreachable from some
  networks (it times out from this machine while `celltrackingchallenge.net`
  works). Download from a university network / VPN and keep the data out of git.
* `--division-containment` is the mitosis knob worth tuning first: it is the
  fraction of a daughter's area that must fall inside the mother mask. On this
dataset the current rule fires zero times out of 6 real divisions, so treat it
  as broken rather than as "needs tuning".
* `--max-gap > 0` produces results the CTC validator rejects (`Label N in
  res_track but not in mask maskXYZ.tif`): a track must be present in *every*
  frame between B and E, and naive gap closing leaves the gap frames empty.
  Anchor the mask or don't use it.
* The two gold annotation sets disagree: on the same cell, `GT/SEG` is ~1.7x
  larger than `GT/TRA` (median 2847 vs 1649 px on seq 01), and `GT/SEG` covers
  only 144 objects across 30 frames. SEG is a small, optimistically-selected
  sample — do not read it as "segmentation solved".

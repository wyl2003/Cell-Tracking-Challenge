"""Figures for the report: segmentation panels and the baseline comparison bars.

    python -m src.make_figure --fig panels --seq data/Fluo-N2DH-GOWT1/01 \
        --res runs/01_RES_cpsam --frame 47 --out figs/cpsam_vs_gt_f047.png

    python -m src.make_figure --fig bars --out figs/model_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.colors import to_rgb

from . import ctc_io


def _random_colors(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.25, 1.0, size=(n + 1, 3))


def _paint(mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """RGBA overlay with one colour per label."""
    labels = np.unique(mask)
    colors = _random_colors(int(labels.max()) + 1)
    rgb = np.zeros((*mask.shape, 3))
    for lab in labels:
        if lab:
            rgb[mask == lab] = colors[lab]
    rgba = np.dstack([rgb, (mask > 0) * alpha])
    return rgba


def _outline(mask: np.ndarray) -> np.ndarray:
    """Boolean edge map of the instance boundaries."""
    edge = np.zeros(mask.shape, dtype=bool)
    edge[:-1, :] |= mask[:-1, :] != mask[1:, :]
    edge[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    edge[-1, :] |= mask[-1, :] != 0
    edge[:, -1] |= mask[:, -1] != 0
    return edge & (mask > 0)


def panels(seq_dir: Path, res_dir: Path, frame: int, out: Path) -> None:
    img = tifffile.imread(seq_dir / f"t{frame:03d}.tif")
    gt = ctc_io.read_mask(seq_dir.with_name(seq_dir.name + "_GT") / "TRA" / f"man_track{frame:03d}.tif")
    pred = ctc_io.read_mask(res_dir / f"mask{frame:03d}.tif")

    fig, axes = plt.subplots(1, 4, figsize=(19, 5.1))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("raw frame %03d" % frame)

    axes[1].imshow(img, cmap="gray")
    axes[1].imshow(_paint(gt))
    axes[1].set_title(f"ground truth (TRA) — {int(gt.max())} cells")

    axes[2].imshow(img, cmap="gray")
    axes[2].imshow(_paint(pred))
    axes[2].set_title(f"Cellpose-SAM zero-shot — {int(pred.max())} cells")

    axes[3].imshow(img, cmap="gray")
    overlay = np.zeros((*gt.shape, 4))
    overlay[_outline(gt)] = [0.1, 0.95, 0.3, 1.0]
    overlay[_outline(pred)] = [1.0, 0.25, 0.2, 1.0]
    axes[3].imshow(overlay)
    axes[3].set_title("GT (green) vs prediction (red)")

    for ax in axes:
        ax.axis("off")
    fig.suptitle("Fluo-N2DH-GOWT1 01 · frame 47 contains a mitosis event (parent 10 → 12 + 14)",
                 fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


RESULTS = {
    # measured with src.diagnostics on this machine (RTX 3060 Laptop, torch cu126)
    "nuclei (zero-shot)": {"01": (0.8573, 0.6662, 0.8565), "02": (0.7563, 0.6827, 0.7550)},
    "cpsam (zero-shot)": {"01": (0.9818, 0.8878, 0.9821), "02": (0.9570, 0.9192, 0.9574)},
}


def bars(out: Path) -> None:
    metrics = ["DET", "SEG", "TRA"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    x = np.arange(len(metrics))
    width = 0.36
    colors = {"nuclei (zero-shot)": "#8fa4b8", "cpsam (zero-shot)": "#2f7fbf"}
    for ax, seq in zip(axes, ("01", "02")):
        for i, (name, vals) in enumerate(RESULTS.items()):
            ax.bar(x + (i - 0.5) * width, vals[seq], width, label=name, color=colors[name])
            for xi, v in zip(x + (i - 0.5) * width, vals[seq]):
                ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_xticks(x, metrics)
        ax.set_ylim(0.5, 1.05)
        ax.set_title(f"sequence {seq}")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("score")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Cellpose 3 `nuclei` vs Cellpose 4 `cpsam`, both zero-shot, same tracker")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fig", choices=["panels", "bars"], required=True)
    ap.add_argument("--seq", default="data/Fluo-N2DH-GOWT1/01")
    ap.add_argument("--res", default="runs/01_RES_cpsam")
    ap.add_argument("--frame", type=int, default=47)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.fig == "panels":
        panels(Path(args.seq), Path(args.res), args.frame, Path(args.out))
    else:
        bars(Path(args.out))


if __name__ == "__main__":
    main()

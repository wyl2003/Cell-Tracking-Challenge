"""Readers/writers for the Cell Tracking Challenge file conventions.

Layout of a CTC sequence (training data)::

    Fluo-N2DL-HeLa/
      01/            t000.tif ... (raw images, 8/16-bit)
      01_GT/
        TRA/         man_track000.tif ... + man_track.txt   (L B E P)
        SEG/         man_seg000.tif ...                     (only segmentation)
      02/ 02_GT/ ...

A result sequence must look like::

    01_RES/
      mask000.tif ...   (instance label images, 16-bit)
      res_track.txt     (L B E P, same columns as man_track.txt)

Column meanings in both tables: L = track label, B = first frame (0-based),
E = last frame, P = parent label (0 = no parent).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import tifffile

_NUM_RE = re.compile(r"(\d+)(?!.*\d)")


def frame_index(path: Path) -> int:
    """Trailing integer of a CTC file name: t001.tif -> 1, man_track012.tif -> 12."""
    m = _NUM_RE.search(path.stem)
    if m is None:
        raise ValueError(f"no frame number in {path.name}")
    return int(m.group(1))


def list_masks(directory: str | Path) -> dict[int, Path]:
    """Map frame index -> mask path, sorted by frame index."""
    directory = Path(directory)
    out = {}
    for p in sorted(directory.glob("*.tif")):
        out[frame_index(p)] = p
    if not out:
        raise FileNotFoundError(f"no .tif masks in {directory}")
    return dict(sorted(out.items()))


def read_mask(path: str | Path) -> np.ndarray:
    return tifffile.imread(str(path)).astype(np.int32)


def write_mask(path: str | Path, mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), mask.astype(np.uint16), compression="zlib")


def read_track_table(path: str | Path) -> np.ndarray:
    """Read man_track.txt / res_track.txt into an (N, 4) int array."""
    path = Path(path)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.replace("\t", " ").split()
        if len(parts) < 4:
            raise ValueError(f"{path}: needs 4 columns 'L B E P', got {line!r}")
        rows.append([int(x) for x in parts[:4]])
    return np.array(rows, dtype=np.int64).reshape(-1, 4)


def write_track_table(path: str | Path, table: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [" ".join(str(int(v)) for v in row) for row in np.asarray(table).reshape(-1, 4)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def renumber_zero_based(src_dir: str | Path, dst_dir: str | Path, prefix: str = "mask") -> dict[int, int]:
    """Copy masks into dst_dir renumbered from 0 (mask000.tif, mask001.tif, ...).

    Some CTC sequences number frames from 1 while the tracking table is 0-based;
    the evaluation tools index masks by their number, so a 0-based copy removes
    an off-by-one without touching any label values. Returns {old: new}.
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    masks = list_masks(src_dir)
    mapping = {old: new for new, old in enumerate(masks)}
    for old, new in mapping.items():
        write_mask(dst_dir / f"{prefix}{new:03d}.tif", read_mask(masks[old]))
    return mapping


def check_consistency(masks: dict[int, np.ndarray], table: np.ndarray) -> list[str]:
    """Cheap sanity checks on a result before spending time on evaluation."""
    problems = []
    labels_in_masks = {int(v) for m in masks.values() for v in np.unique(m) if v != 0}
    labels_in_table = {int(r[0]) for r in table}
    if labels_in_masks - labels_in_table:
        problems.append(f"labels in masks but not in res_track.txt: {sorted(labels_in_masks - labels_in_table)}")
    if labels_in_table - labels_in_masks:
        problems.append(f"labels in res_track.txt but never in masks: {sorted(labels_in_table - labels_in_masks)}")
    frames = sorted(masks)
    for label, begin, end, parent in table:
        if begin > end:
            problems.append(f"track {label}: begin {begin} > end {end}")
        if parent and parent not in labels_in_table:
            problems.append(f"track {label}: parent {parent} does not exist")
        if parent and parent == label:
            problems.append(f"track {label}: is its own parent")
        present = [f for f in frames if np.any(masks[f] == label)]
        if not present:
            continue
        if min(present) != begin or max(present) != end:
            problems.append(
                f"track {label}: table says [{begin},{end}] but appears in [{min(present)},{max(present)}]"
            )
    return problems

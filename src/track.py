"""A small, readable tracking-by-detection baseline for CTC sequences.

Input: one instance mask per frame (from any segmenter, or the gold SEG masks).
Output: CTC relabelled masks + a lineage table (L B E P).

Deliberately simple so it can serve as the "baseline" row of an ablation table:
  1. frame-to-frame association by IoU + Hungarian assignment
  2. optional gap closing for `max_gap` frames
  3. mitosis detection: when one object becomes two, the mother track ends and
     two child tracks start (CTC scores lineage separately from segmentation,
     and a wrong division edge costs 1.5 AOGM units on top of the node errors)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def frame_stats(mask: np.ndarray) -> dict[int, dict]:
    """Per-label area and centroid for one frame."""
    out = {}
    for label in np.unique(mask):
        if label == 0:
            continue
        ys, xs = np.nonzero(mask == label)
        out[int(label)] = {"area": float(len(ys)), "cy": float(ys.mean()), "cx": float(xs.mean())}
    return out


def iou_mask_label(mask_a: np.ndarray, label_a: int, mask_b: np.ndarray, label_b: int) -> float:
    a, b = mask_a == label_a, mask_b == label_b
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def track_sequence(
    masks: dict[int, np.ndarray],
    iou_threshold: float = 0.2,
    max_gap: int = 0,
    detect_divisions: bool = True,
    division_containment: float = 0.5,
    area_ratio_range: tuple[float, float] = (0.5, 1.8),
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Track instance masks across frames.

    Returns (relabelled masks keyed by original frame index, track table (N, 4)).
    """
    frames = sorted(masks)
    tracks: dict[int, dict] = {}          # track -> {begin, end, parent}
    frame_map: dict[int, dict[int, int]] = {}   # frame -> {mask_label: track}
    next_label = 1

    active: dict[int, int] = {}           # track -> last frame it was seen
    last_mask_label: dict[int, int] = {}  # track -> mask label in that frame

    def new_track(frame: int, mask_label: int, parent: int = 0) -> int:
        nonlocal next_label
        track = next_label
        next_label += 1
        tracks[track] = {"begin": frame, "end": frame, "parent": parent}
        active[track] = frame
        last_mask_label[track] = mask_label
        frame_map.setdefault(frame, {})[mask_label] = track
        return track

    for frame in frames:
        cur = masks[frame]
        cur_labels = [int(v) for v in np.unique(cur) if v != 0]
        cur_areas = {l: float((cur == l).sum()) for l in cur_labels}
        taken: set[int] = set()

        # ---- 1. continue existing tracks (newest frame first, so gaps can close)
        linked: list[tuple[int, int, int]] = []  # (track, prev_frame, mask_label_now)
        for gap in range(max_gap + 1):
            prev_frame = frame - 1 - gap
            if prev_frame not in masks:
                continue
            candidates = {
                t: last_mask_label[t]
                for t, last in active.items()
                if last == prev_frame and t not in frame_map.get(frame, {})
            }
            remaining = [l for l in cur_labels if l not in taken]
            if not candidates or not remaining:
                continue
            keys = list(candidates)
            scores = np.array(
                [[iou_mask_label(masks[prev_frame], candidates[t], cur, l) for l in remaining] for t in keys]
            )
            rows, cols = linear_sum_assignment(-scores)
            for r, c in zip(rows, cols):
                if scores[r, c] < iou_threshold:
                    continue
                track, mask_label = keys[r], remaining[c]
                tracks[track]["end"] = frame
                active[track] = frame
                last_mask_label[track] = mask_label
                frame_map.setdefault(frame, {})[mask_label] = track
                taken.add(mask_label)
                linked.append((track, prev_frame, mask_label))

        # ---- 2. mitosis: a "continuation" that is really a split becomes 1 -> 2
        if detect_divisions:
            for track, prev_frame, mask_label in linked:
                if prev_frame != frame - 1:
                    continue
                p_mask = masks[prev_frame] == last_mask_label_before(track, prev_frame, frame_map, masks)
                p_area = float(p_mask.sum())
                if p_area == 0:
                    continue
                siblings = []
                for other in cur_labels:
                    if other == mask_label or other in taken:
                        continue
                    inter = float(np.logical_and(p_mask, cur == other).sum())
                    if cur_areas[other] and inter / cur_areas[other] >= division_containment:
                        siblings.append((inter, other))
                if not siblings:
                    continue
                siblings.sort(reverse=True)
                inter, sibling = siblings[0]
                inter_main = float(np.logical_and(p_mask, cur == mask_label).sum())
                child_area = cur_areas[mask_label] + cur_areas[sibling]
                if not (area_ratio_range[0] <= child_area / p_area <= area_ratio_range[1]):
                    continue
                if inter_main / cur_areas[mask_label] < division_containment:
                    continue
                # rewind the mother track and start two children
                frame_map[frame].pop(mask_label, None)
                tracks[track]["end"] = prev_frame
                active.pop(track, None)
                taken.discard(mask_label)
                new_track(frame, mask_label, parent=track)
                new_track(frame, sibling, parent=track)
                taken.update({mask_label, sibling})

        # ---- 3. everything still unassigned starts a new track
        for mask_label in cur_labels:
            if mask_label not in taken:
                new_track(frame, mask_label)

    relabelled = {}
    for frame in frames:
        out = np.zeros_like(masks[frame], dtype=np.int32)
        for mask_label, track in frame_map.get(frame, {}).items():
            out[masks[frame] == mask_label] = track
        relabelled[frame] = out

    table = np.array(
        [[t, info["begin"], info["end"], info["parent"]] for t, info in sorted(tracks.items())],
        dtype=np.int64,
    ).reshape(-1, 4)
    return relabelled, table


def last_mask_label_before(track: int, prev_frame: int, frame_map, masks) -> int:
    """Mask label of `track` in `prev_frame` (recovered from the frame map)."""
    for mask_label, t in frame_map.get(prev_frame, {}).items():
        if t == track:
            return mask_label
    raise KeyError(f"track {track} has no label in frame {prev_frame}")

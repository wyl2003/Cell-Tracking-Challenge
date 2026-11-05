"""CTC tracking with gated assignment and optional motion/division hypotheses.

Default: adjacent-frame IoU association and overlap-based division detection.
Experimental switches: constant-velocity association and geometric divisions.
No missing masks are invented: gaps terminate tracks, and max_gap > 0 is rejected.
The unchanged original implementation remains in ../track.py for comparisons.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment


def frame_stats(mask: np.ndarray) -> dict[int, dict]:
    out = {}
    for label in np.unique(mask):
        if label:
            ys, xs = np.nonzero(mask == label)
            out[int(label)] = {"area": float(len(ys)), "cy": float(ys.mean()), "cx": float(xs.mean())}
    return out


def iou_mask_label(mask_a, label_a, mask_b, label_b) -> float:
    a, b = mask_a == label_a, mask_b == label_b
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return float(inter / union) if union else 0.0


def overlap_matrix(previous, current, previous_labels, current_labels):
    """Compute intersections in one image pass, without indexing by raw IDs."""
    pa, pi = np.unique(previous, return_inverse=True)
    ca, ci = np.unique(current, return_inverse=True)
    counts = np.bincount(pi.ravel() * len(ca) + ci.ravel(),
                         minlength=len(pa) * len(ca)).reshape(len(pa), len(ca))
    return counts[np.ix_(np.searchsorted(pa, previous_labels),
                         np.searchsorted(ca, current_labels))].astype(float)


def gated_assignment(scores: np.ndarray, valid: np.ndarray):
    """Maximize valid match cardinality, then score; allow unmatched rows.

    Gating *after* Hungarian can discard a forbidden edge that already displaced
    a valid alternative. Dummy columns and pre-gating avoid that failure.
    """
    n, m = scores.shape
    if not n or not m:
        return []
    reward = np.full((n, m + n), -1e9, dtype=float)
    reward[:, :m] = np.where(valid, min(n, m) + 1 + scores, -1e9)
    reward[:, m:] = 0
    rows, cols = linear_sum_assignment(-reward)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if c < m and valid[r, c]]


def validate_masks(masks):
    if not masks:
        raise ValueError("at least one frame is required")
    frames = sorted(masks)
    if any(not isinstance(f, (int, np.integer)) or f < 0 for f in frames):
        raise ValueError("frame indices must be nonnegative integers")
    if frames != list(range(frames[0], frames[-1] + 1)):
        raise ValueError("frames must be consecutive; sparse GT/SEG is not a tracking sequence")
    shape = np.asarray(masks[frames[0]]).shape
    for f, mask in masks.items():
        if mask.ndim != 2 or mask.shape != shape or mask.size == 0:
            raise ValueError(f"frame {f}: expected nonempty 2D masks of identical shape")
        if not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0):
            raise ValueError(f"frame {f}: masks must contain nonnegative integer labels")


def track_sequence(masks: dict[int, np.ndarray], iou_threshold: float = 0.2,
                   max_gap: int = 0, detect_divisions: bool = True,
                   division_containment: float = 0.5,
                   area_ratio_range: tuple[float, float] = (0.5, 1.8),
                   association: str = "iou", max_distance: float = 30.0,
                   division_mode: str = "overlap", division_distance: float = 2.0):
    """Return masks and L/B/E/P table using the original frame numbers.

    hybrid: 0.6 predicted-mask IoU + 0.3 proximity to velocity prediction + 0.1 area
    similarity; motion matches require distance <= max_distance and area ratio
    in [0.5, 2]. max_distance is pixels/frame; tune on development data only.

    motion division: two daughters near the predicted mother, each 20--80% of
    mother area, balanced within 3:1, total area within area_ratio_range; their
    area-weighted centre must be within 0.75 mother radii of the prediction.
    This is a hypothesis generator, not evidence of improved biological recall.
    """
    validate_masks(masks)
    if max_gap != 0:
        raise ValueError("max_gap > 0 is unsupported: CTC requires a mask in every track frame; "
                         "use max_gap=0 until mask propagation is implemented and evaluated")
    if association not in {"iou", "hybrid"} or division_mode not in {"overlap", "motion"}:
        raise ValueError("unknown association or division mode")
    if not 0 <= iou_threshold <= 1 or not 0 <= division_containment <= 1:
        raise ValueError("IoU and containment thresholds must lie in [0, 1]")
    if not np.isfinite(max_distance) or max_distance <= 0 or not np.isfinite(division_distance) or division_distance <= 0:
        raise ValueError("distance gates must be finite and positive")
    if not 0 < area_ratio_range[0] <= area_ratio_range[1] < np.inf:
        raise ValueError("invalid division area ratio range")

    frames = sorted(masks)
    stats = {f: frame_stats(masks[f]) for f in frames}
    tracks, histories, maps = {}, {}, {}
    next_id = 1

    def centre(s):
        return np.array([s['cy'], s['cx']])

    def start(f, label, parent=0):
        nonlocal next_id
        tid = next_id
        next_id += 1
        tracks[tid] = [tid, f, f, parent]
        maps[f][label] = tid
        histories[tid] = [centre(stats[f][label])]

    for f in frames:
        maps[f] = {}
        current = sorted(stats[f])
        if f == frames[0] or not stats[f - 1] or not current:
            for label in current:
                start(f, label)
            continue
        previous = sorted(stats[f - 1])
        tids = [maps[f - 1][l] for l in previous]
        pa = np.array([stats[f - 1][l]['area'] for l in previous])
        ca = np.array([stats[f][l]['area'] for l in current])
        pc = np.array([centre(stats[f - 1][l]) for l in previous])
        cc = np.array([centre(stats[f][l]) for l in current])
        predicted = pc.copy()
        for i, tid in enumerate(tids):
            h = histories[tid]
            if len(h) >= 2:
                predicted[i] += h[-1] - h[-2]
        inter = overlap_matrix(masks[f - 1], masks[f], previous, current)
        iou = inter / (pa[:, None] + ca[None, :] - inter)
        distance = np.linalg.norm(predicted[:, None, :] - cc[None, :, :], axis=2)
        ratio = ca[None, :] / pa[:, None]
        valid = (iou >= iou_threshold) & (inter > 0)
        scores = iou.copy()
        if association == 'hybrid':
            # Compare current cells with translated masks, not stale locations.
            # Otherwise high raw IoU swaps identities when two cells cross.
            predicted_inter = inter.copy()
            predicted_area = pa.copy()
            for r, label in enumerate(previous):
                displacement = np.rint(predicted[r] - pc[r]).astype(int)
                ys, xs = np.nonzero(masks[f - 1] == label)
                ys, xs = ys + displacement[0], xs + displacement[1]
                inside = (ys >= 0) & (xs >= 0) & (ys < masks[f].shape[0]) & (xs < masks[f].shape[1])
                predicted_area[r] = np.count_nonzero(inside)
                hit_labels, counts = np.unique(masks[f][ys[inside], xs[inside]], return_counts=True)
                hits = dict(zip(hit_labels, counts))
                predicted_inter[r] = [hits.get(l, 0) for l in current]
            predicted_iou = predicted_inter / (predicted_area[:, None] + ca[None, :] - predicted_inter)
            valid = ((predicted_iou >= iou_threshold) & (predicted_inter > 0)) | (
                (distance <= max_distance) & (ratio >= 0.5) & (ratio <= 2.0))
            scores = 0.6 * predicted_iou + 0.3 * np.exp(-distance / max_distance) + 0.1 * np.minimum(ratio, 1 / ratio)
        matches = dict(gated_assignment(scores, valid))
        owners = {c: r for r, c in matches.items()}
        proposals = []
        if detect_divisions:
            for r in range(len(previous)):
                available = [c for c in range(len(current)) if c not in owners or owners[c] == r]
                for a, b in combinations(available, 2):
                    total = ca[a] + ca[b]
                    if not area_ratio_range[0] <= total / pa[r] <= area_ratio_range[1]:
                        continue
                    if division_mode == 'overlap':
                        # Preserve the old model's prerequisite: one daughter linked.
                        if matches.get(r) not in (a, b):
                            continue
                        contain = min(inter[r, a] / ca[a], inter[r, b] / ca[b])
                        if contain < division_containment or min(inter[r, a], inter[r, b]) == 0:
                            continue
                        quality = contain
                    else:
                        radius = np.sqrt(pa[r] / np.pi)
                        if not (0.2 <= ca[a] / pa[r] <= 0.8 and 0.2 <= ca[b] / pa[r] <= 0.8):
                            continue
                        if min(ca[a], ca[b]) / max(ca[a], ca[b]) < 1 / 3:
                            continue
                        if max(distance[r, a], distance[r, b]) > division_distance * radius:
                            continue
                        midpoint = (ca[a] * cc[a] + ca[b] * cc[b]) / total
                        offset = np.linalg.norm(midpoint - predicted[r]) / radius
                        if offset > 0.75:
                            continue
                        quality = 1 / (1 + offset + abs(np.log(total / pa[r])))
                    proposals.append((-quality, r, a, b))
        split_rows, used = set(), set()
        for _, r, a, b in sorted(proposals):
            if r in split_rows or a in used or b in used:
                continue
            # Do not discard an unrelated continuation of this mother.
            if r in matches and matches[r] not in (a, b):
                continue
            start(f, current[a], tids[r])
            start(f, current[b], tids[r])
            split_rows.add(r)
            used.update((a, b))
        for r, c in matches.items():
            if r in split_rows or c in used:
                continue
            tid = tids[r]
            maps[f][current[c]] = tid
            tracks[tid][2] = f
            histories[tid].append(cc[c])
            histories[tid] = histories[tid][-2:]
            used.add(c)
        for c, label in enumerate(current):
            if c not in used:
                start(f, label)

    relabelled = {}
    for f in frames:
        out = np.zeros_like(masks[f], dtype=np.int32)
        for label, tid in maps[f].items():
            out[masks[f] == label] = tid
        relabelled[f] = out
    return relabelled, np.asarray(list(tracks.values()), dtype=np.int64).reshape(-1, 4)

"""Conservative temporal missing-cell recovery for CTC instance masks.

Uses no ground truth. Preserves original multi-scale instance labels/pixels.
Requires numpy, scipy, tifffile; project merge_multiscale_masks is used when possible.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi


def files_in(path):
    path = Path(path)
    files = sorted(path.glob('mask[0-9][0-9][0-9].tif'))
    if not files:
        raise ValueError(f'No mask###.tif files in {path}')
    return files


def read_masks(path):
    files = files_in(path)
    frame_ids = [int(f.stem[4:]) for f in files]
    if frame_ids != list(range(len(files))):
        raise ValueError(f'Non-consecutive mask frame indices in {path}: {frame_ids[:5]} ...')
    masks = [tifffile.imread(f) for f in files]
    if any(x.ndim != 2 or x.shape != masks[0].shape for x in masks):
        raise ValueError(f'Inconsistent mask dimensions in {path}')
    return masks


def merge(baseline, small, overlap_threshold=.10, min_area=200):
    if baseline.shape != small.shape:
        raise ValueError('Different shapes for baseline and auxiliary masks')
    result = baseline.copy()
    occupied = baseline > 0
    next_id = int(baseline.max()) + 1
    for label in np.unique(small):
        if label == 0:
            continue
        candidate = small == label
        area = int(candidate.sum())
        if area < min_area:
            continue
        overlap = np.count_nonzero(candidate & occupied) / area
        if overlap > overlap_threshold:
            continue
        new_region = candidate & ~occupied
        if int(new_region.sum()) < min_area:
            continue
        result[new_region] = next_id
        next_id += 1
    return result


def props(mask, min_area):
    objects = []
    for lab in np.unique(mask):
        if lab == 0:
            continue
        yy, xx = np.nonzero(mask == lab)
        if len(yy) < min_area:
            continue
        objects.append({'id':int(lab),'cy':float(yy.mean()), 'cx':float(xx.mean()), 'area':len(yy)})
    return objects


def shift_mask(binary, dy, dx):
    # integer translation without wraparound
    h, w = binary.shape
    dy, dx = int(round(dy)), int(round(dx))
    output = np.zeros_like(binary)
    sy0, sy1 = max(0, -dy), min(h, h-dy)
    sx0, sx1 = max(0, -dx), min(w, w-dx)
    if sy1 > sy0 and sx1 > sx0:
        output[sy0+dy:sy1+dy, sx0+dx:sx1+dx] = binary[sy0:sy1, sx0:sx1]
    return output


def robust_contrast(img, proposal, ring_width=8):
    ring = ndi.binary_dilation(proposal, iterations=ring_width) & ~ndi.binary_dilation(proposal, iterations=2)
    if not np.any(ring):
        return float('nan')
    inside = float(np.median(img[proposal]))
    outside = float(np.median(img[ring]))
    # Handle images containing many zero intensities without divide-by-zero.
    return (inside + 1.0) / (outside + 1.0)


def recover_frame(prev, cur, nex, img, cfg):
    ps = props(prev, cfg.min_area)
    ns = props(nex, cfg.min_area)
    cs = props(cur, cfg.min_area)
    result = cur.copy()
    events = []
    proposals = []
    # Each next-frame object can support at most one recovery.
    pairs = []
    for a in ps:
        for b in ns:
            dist = float(np.hypot(a['cy']-b['cy'], a['cx']-b['cx']))
            ratio = min(a['area'], b['area']) / max(a['area'], b['area'])
            if dist <= cfg.max_two_frame_motion and ratio >= cfg.min_area_ratio:
                pairs.append((dist, a, b))
    pairs.sort(key=lambda x: x[0])
    used_p, used_n = set(), set()
    for dist, a, b in pairs:
        if a['id'] in used_p or b['id'] in used_n:
            continue
        # When there's an existing cell around predicted center, it isn't a missing cell.
        cy, cx = (a['cy']+b['cy'])/2, (a['cx']+b['cx'])/2
        closest = min((np.hypot(c['cy']-cy, c['cx']-cx) for c in cs), default=float('inf'))
        if closest <= cfg.existing_center_radius:
            continue
        p_mask = prev == a['id']
        n_mask = nex == b['id']
        p_shift = shift_mask(p_mask, cy-a['cy'], cx-a['cx'])
        n_shift = shift_mask(n_mask, cy-b['cy'], cx-b['cx'])
        union = p_shift | n_shift
        intersection = p_shift & n_shift
        if not np.any(union) or intersection.sum()/union.sum() < cfg.min_shape_iou:
            continue
        # Conservative common-shape estimate.
        proposal = intersection
        if int(proposal.sum()) < cfg.min_area:
            continue
        occupancy = np.count_nonzero(proposal & (result > 0)) / proposal.sum()
        if occupancy > cfg.max_occupied_fraction:
            continue
        proposal = proposal & (result == 0)
        if int(proposal.sum()) < cfg.min_area:
            continue
        contrast = robust_contrast(img, proposal)
        if not np.isfinite(contrast) or contrast < cfg.min_contrast:
            continue
        proposals.append((proposal, a, b, contrast, dist))
        used_p.add(a['id'])
        used_n.add(b['id'])
    new_id = int(cur.max()) + 1
    for proposal, a, b, contrast, dist in proposals:
        # Recheck overlap after preceding proposals have been inserted.
        proposal = proposal & (result == 0)
        if proposal.sum() < cfg.min_area:
            continue
        result[proposal] = new_id
        events.append({'new_id':new_id,'prev_id':a['id'],'next_id':b['id'],
                       'area':int(proposal.sum()),'contrast_ratio':round(contrast,3),
                       'two_frame_distance':round(dist,2)})
        new_id += 1
    return result, events


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', required=True)
    p.add_argument('--small', help='Second-scale masks; needed unless --merged supplied')
    p.add_argument('--merged', help='Already-merged multi-scale masks')
    p.add_argument('--images', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--min-area', type=int, default=200)
    p.add_argument('--max-two-frame-motion', type=float, default=24)
    p.add_argument('--existing-center-radius', type=float, default=15)
    p.add_argument('--min-area-ratio', type=float, default=.60)
    p.add_argument('--min-shape-iou', type=float, default=.50)
    p.add_argument('--max-occupied-fraction', type=float, default=.10)
    p.add_argument('--min-contrast', type=float, default=1.15)
    p.add_argument('--inspect-only', action='store_true', help='Report frames/inputs without writing output')
    cfg = p.parse_args()
    baseline = read_masks(cfg.baseline)
    if cfg.merged:
        multi = read_masks(cfg.merged)
        mode = 'existing merged masks'
    else:
        if not cfg.small:
            p.error('Provide either --merged or --small')
        small = read_masks(cfg.small)
        if len(small) != len(baseline):
            raise ValueError('Baseline and auxiliary frame counts differ')
        multi = [merge(a, b, min_area=cfg.min_area) for a, b in zip(baseline, small)]
        mode = 'merged baseline+auxiliary'
    if len(multi) != len(baseline):
        raise ValueError('Baseline and multi-scale frame counts differ')
    img_dir = Path(cfg.images)
    img_files = [img_dir / f't{i:03d}.tif' for i in range(len(multi))]
    missing = [str(f) for f in img_files if not f.is_file()]
    if missing:
        raise ValueError(f'Missing images: {missing[:3]}')
    if any(b.shape != m.shape for b, m in zip(baseline, multi)):
        raise ValueError('Baseline and multi-scale image dimensions differ')
    new_labels = [int(len(set(np.unique(m))-set(np.unique(b))-{0})) for b,m in zip(baseline,multi)]
    # Check that merged masks preserve baseline, including ID values.
    if any(not np.array_equal(m[b!=0], b[b!=0]) for b,m in zip(baseline,multi)):
        raise ValueError('Multi-scale masks overwrite baseline pixels/IDs; investigate before proceeding')
    print(f'Input mode: {mode}; {len(multi)} frames; shape {multi[0].shape}; added-label count {sum(new_labels)}')
    if cfg.inspect_only:
        return
    out = Path(cfg.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Output folder not empty: {out}. Use a new folder to preserve prior results.')
    out.mkdir(parents=True, exist_ok=True)
    all_events = []
    added = 0
    for i, mask in enumerate(multi):
        if 0 < i < len(multi)-1:
            img = tifffile.imread(img_files[i])
            if img.ndim != 2 or img.shape != mask.shape:
                raise ValueError(f'Image size/type mismatch: {img_files[i]}')
            new, events = recover_frame(multi[i-1], mask, multi[i+1], img, cfg)
        else:
            new, events = mask.copy(), []
        for e in events:
            e['frame'] = i
            all_events.append(e)
        added += len(events)
        tifffile.imwrite(out/f'mask{i:03d}.tif', new)
    with (out/'temporal_events.json').open('w',encoding='utf-8') as f:
        json.dump({'parameters':vars(cfg), 'input_mode':mode,'events':all_events},f,indent=2)
    print(f'Temporal recovered instances: {added} in {len(set(e["frame"] for e in all_events))} frames')
    print(f'Masks: {out}; events: {out/"temporal_events.json"}')


if __name__ == '__main__':
    main()

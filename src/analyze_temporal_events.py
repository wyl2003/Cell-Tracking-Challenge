"""Visualize and ablate each temporal recovery event without modifying originals.

Run from project root: python -m src.analyze_temporal_events ...
Dependencies: numpy, tifffile, matplotlib, scipy (existing project environment).
"""
import argparse
import json
import os
import shutil
from pathlib import Path
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage as ndi


def read_sequence(directory):
    directory = Path(directory)
    files = sorted(directory.glob('mask[0-9][0-9][0-9].tif'))
    if not files:
        raise ValueError(f'No mask###.tif files in {directory}')
    nums = [int(p.stem[4:]) for p in files]
    if nums != list(range(len(files))):
        raise ValueError(f'Frames are missing in {directory}')
    return [tifffile.imread(p) for p in files]


def read_image(path):
    image = tifffile.imread(path)
    if image.ndim != 2:
        raise ValueError(f'Expected 2D microscopy image: {path}')
    return image


def show_event(image, before, after, ev, path, padding):
    label = int(ev['new_id'])
    newly_added = (after == label) & (before == 0)
    ys, xs = np.where(newly_added)
    if not len(ys):
        raise ValueError(f"Event has no new region: {ev}")
    h,w=image.shape
    y0=max(0, int(ys.min())-padding); y1=min(h, int(ys.max())+padding+1)
    x0=max(0, int(xs.min())-padding); x1=min(w, int(xs.max())+padding+1)
    im=image[y0:y1,x0:x1]
    base=(before[y0:y1,x0:x1]>0)
    new=newly_added[y0:y1,x0:x1]
    fig, axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    lo,hi=np.percentile(im,[1,99.5]); hi=max(hi,lo+1)
    for ax in axes:
        ax.imshow(im,cmap='gray',vmin=lo,vmax=hi)
        ax.set_axis_off()
    axes[0].set_title('Original microscopy')
    axes[1].contour(base.astype(float),levels=[.5],colors=['cyan'],linewidths=1)
    axes[1].set_title('Before: existing instances')
    axes[2].contour(base.astype(float),levels=[.5],colors=['cyan'],linewidths=1)
    axes[2].contour(new.astype(float),levels=[.5],colors=['orange'],linewidths=1.6)
    axes[2].set_title('After: orange=temporal addition')
    fig.suptitle(f"Frame {int(ev['frame']):03d}, instance {label}, area {int(new.sum())}")
    fig.savefig(path,dpi=150)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--temporal',required=True,help='Directory containing temporal mask###.tif and temporal_events.json')
    ap.add_argument('--images',required=True,help='Original microscopy t###.tif directory')
    ap.add_argument('--out',required=True,help='New output directory for plots and ablation masks')
    ap.add_argument('--padding',type=int,default=40)
    args=ap.parse_args()
    src=Path(args.temporal); out=Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Output must be empty/nonexistent: {out}')
    with (src/'temporal_events.json').open(encoding='utf-8') as f:
        data=json.load(f)
    events=data.get('events',[])
    if not events:
        raise ValueError('No events to analyze')
    final=read_sequence(src)
    n=len(final)
    names=[src/f'mask{i:03d}.tif' for i in range(n)]
    for i,ev in enumerate(events):
        fr=int(ev['frame']); lab=int(ev['new_id'])
        if fr<0 or fr>=n or not np.any(final[fr]==lab):
            raise ValueError(f'Event label missing, frame={fr} ID={lab}')
        if sum(int(e['frame'])==fr and int(e['new_id'])==lab for e in events)>1:
            raise ValueError('Duplicate event label in JSON')
    # Reconstruct multi-scale-only by stripping exclusively temporal-added labels.
    base=[]
    byframe={}
    for e in events:
        byframe.setdefault(int(e['frame']),[]).append(int(e['new_id']))
    for fr,mask in enumerate(final):
        b=mask.copy()
        for lab in byframe.get(fr,[]):
            b[b==lab]=0
        base.append(b)
    # NOTE: Each label must be temporal-only; inspect before relying on ablation.
    out.mkdir(parents=True)
    (out/'images').mkdir()
    case_info=[]
    for k, ev in enumerate(events,1):
        fr=int(ev['frame']); lab=int(ev['new_id'])
        imgpath=Path(args.images)/f't{fr:03d}.tif'
        img=read_image(imgpath)
        if img.shape!=final[fr].shape:
            raise ValueError(f'Image-mask shape mismatch: {imgpath}')
        plotpath=out/'images'/f'event{k:02d}_frame{fr:03d}.png'
        show_event(img,base[fr],final[fr],ev,plotpath,args.padding)
        variant=out/f'without_event{k:02d}'
        variant.mkdir()
        for i,mask in enumerate(final):
            # Only remove exactly one event, leave all other temporal events unchanged.
            if i==fr:
                m=mask.copy(); m[m==lab]=0
                tifffile.imwrite(variant/f'mask{i:03d}.tif',m)
            else:
                # Hard-link unchanged masks (Windows same-drive NTFS) to avoid large copies.
                # Never edit linked files in place. Fall back to copying if unavailable.
                target=variant/f'mask{i:03d}.tif'
                try:
                    os.link(names[i], target)
                except OSError:
                    shutil.copy2(names[i], target)
        case_info.append({'event':k,'frame':fr,'new_id':lab,'area':int((final[fr]==lab).sum()),
                          'without_directory':str(variant),'preview':str(plotpath),
                          'note':'Compare full temporal score vs variant score; lower variant score means the event helped.'})
    baseline_dir=out/'multiscale_only'
    baseline_dir.mkdir()
    for i,mask in enumerate(base):
        tifffile.imwrite(baseline_dir/f'mask{i:03d}.tif',mask)
    with (out/'event_ablation_manifest.json').open('w',encoding='utf-8') as f:
        json.dump({'temporal_directory':str(src),'n_frames':n,
                   'multiscale_only':str(baseline_dir),'cases':case_info},f,indent=2)
    print(f'Prepared {len(events)} event previews + leave-one-out variants under: {out}')
    for c in case_info:
        print(f"Event {c['event']:02d}: frame {c['frame']:03d}, ID {c['new_id']}, area {c['area']} pixels; image {c['preview']}")
    print('IMPORTANT: Compare multiscale_only against your prior Multi-scale outputs before interpreting ablations.')


if __name__=='__main__':
    main()

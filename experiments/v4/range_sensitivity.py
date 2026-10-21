"""Paired BEV forward-20-m sensitivity test for V4 time-free Path model.

Run from bev_trajectory_demo root. Does not retrain or overwrite weights.
The BEV cut is an ego-frame x <= 20 m half-plane: rear-visible geometry is retained.
Reference/navigation/GT remain exactly identical across both input conditions.
"""
import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import BEVDataset
from models import TrajectoryModel

MODES = ('reference_markings', 'full')
NAVS = ('straight', 'left', 'right')


def clip_polyline_xmax(points, xmax, eps=1e-7):
    """Clip ordered polyline to x <= xmax; interpolate exact boundary intersections.

    Return separate polylines if the original exits and subsequently re-enters.
    No fake segments joining across invisible geometry are created.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return []
    fragments, current = [], []

    def append(p):
        if not current or np.linalg.norm(p - current[-1]) > eps:
            current.append(p.copy())

    def finish():
        if len(current) >= 2 and sum(np.linalg.norm(b-a) for a, b in zip(current, current[1:])) > 1e-3:
            fragments.append(np.asarray(current).round(5).tolist())
        current.clear()

    for a, b in zip(pts[:-1], pts[1:]):
        inside_a = a[0] <= xmax + eps
        inside_b = b[0] <= xmax + eps
        if inside_a and inside_b:
            append(a)
            append(b)
        elif inside_a and not inside_b:
            append(a)
            t = (xmax - a[0]) / (b[0] - a[0])
            append(a + t * (b - a))
            finish()
        elif not inside_a and inside_b:
            t = (xmax - a[0]) / (b[0] - a[0])
            append(a + t * (b - a))
            append(b)
        else:
            # Both endpoints outside x<=xmax: nothing visible on a straight segment.
            finish()
    finish()
    return fragments


def clip_elements(elements, xmax):
    output = []
    for element in elements:
        for points in clip_polyline_xmax(element['points'], xmax):
            item = dict(element)
            item['points'] = points
            output.append(item)
    return output


class ClippedDataset(BEVDataset):
    def __init__(self, *args, forward_m=20., **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_m = float(forward_m)
        new_rows = []
        for r in self.rows:
            r2 = dict(r)
            r2['lane_markings'] = clip_elements(r['lane_markings'], self.forward_m)
            r2['road_boundaries'] = clip_elements(r['road_boundaries'], self.forward_m)
            new_rows.append(r2)
        self.rows = new_rows


def mask_inputs(batch, mode):
    """Identical masking rule to V4 ablation.py."""
    if mode == 'full':
        return batch
    valid = batch['line_valid'].clone()
    is_marking = batch['lines'][:, :, :, 5].amax(dim=-1) > 0.5
    valid &= is_marking
    out = dict(batch)
    out['line_valid'] = valid
    out['lines'] = batch['lines'] * valid[:, :, None, None].to(batch['lines'].dtype)
    return out


def evaluate_model(ckpt_path, mode, dataset, args):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if ckpt.get('model_type') != 'arc_length_path_v4':
        raise ValueError(f'Not a V4 path checkpoint: {ckpt_path}')
    if ckpt.get('ablation_mode', 'full') != mode:
        raise ValueError(f'Checkpoint mode does not match {mode}: {ckpt_path}')
    if ckpt.get('seed', args.seed) != args.seed:
        raise ValueError(f'Checkpoint seed mismatch: {ckpt_path}')
    if ckpt['horizon'] != dataset.path_points:
        raise ValueError('Checkpoint horizon differs from dataset GT path length')
    model = TrajectoryModel(horizon=ckpt['horizon']).to(args.device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    predictions = []
    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers)
    with torch.inference_mode():
        for raw in dl:
            batch = mask_inputs({k: v.to(args.device) for k, v in raw.items()}, mode)
            predictions.append(model(batch).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def summarize(error, indices):
    chosen = error[indices]
    if len(chosen) == 0:
        return None
    per_path = chosen.mean(axis=1)
    return {'count': len(chosen), 'path_mean_m': float(per_path.mean()),
            'endpoint_m': float(chosen[:, -1].mean()),
            'p95_scene_mean_m': float(np.percentile(per_path, 95))}


def plot_curves(errors, nav, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    x = np.arange(next(iter(errors.values())).shape[1])
    for ax, group in zip(axes, ('all', 'turns')):
        inds = np.arange(len(nav)) if group == 'all' else np.where(nav != 0)[0]
        for key, values in errors.items():
            ax.plot(x, values[inds].mean(axis=0), label=key, lw=1.8)
        ax.set(xlabel='Path arc length (m)', title=f'{group}: mean position error')
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Position error (m)')
    fig.tight_layout()
    fig.savefig(out / 'error_vs_distance.png', dpi=170)
    plt.close(fig)


def plot_scenes(full_ds, clipped_ds, preds, errors, nav, forward, out):
    # Choose representative scene per category: median change in path error for FULL model.
    choices = []
    full_delta = errors['full_20m'].mean(axis=1) - errors['full_original'].mean(axis=1)
    for n in range(3):
        idxs = np.where(nav == n)[0]
        if len(idxs):
            choices.append(int(idxs[np.argmin(np.abs(full_delta[idxs] - np.median(full_delta[idxs])))]))
    fig, axes = plt.subplots(1, len(choices), figsize=(6*len(choices), 6), squeeze=False)
    for ax, i in zip(axes[0], choices):
        original, cropped = full_ds.rows[i], clipped_ds.rows[i]
        for kind, color, label in (('lane_markings', '#b9ab66', 'BEV markings'),
                                    ('road_boundaries', '#888888', 'Road boundary')):
            for j, element in enumerate(original[kind]):
                p = np.asarray(element['points'])
                ax.plot(p[:, 0], p[:, 1], color=color, alpha=.17, lw=1.2,
                        label=f'{label} original' if j == 0 else None)
            for j, element in enumerate(cropped[kind]):
                p = np.asarray(element['points'])
                ax.plot(p[:, 0], p[:, 1], color=color, alpha=.85, lw=1.4,
                        label=f'{label} clipped' if j == 0 else None)
        ref = np.asarray(original['reference_line'])
        gt = np.asarray(original['gt_path'])
        ax.plot(ref[:, 0], ref[:, 1], '--', color='orange', alpha=.7, label='Biased reference')
        ax.plot(gt[:, 0], gt[:, 1], color='green', lw=3, label='GT path')
        for key, color, ls in (('reference_markings_original', 'blue', '-'),
                               ('reference_markings_20m', 'blue', '--'),
                               ('full_original', 'red', '-'),
                               ('full_20m', 'red', '--')):
            p = preds[key][i]
            ax.plot(p[:, 0], p[:, 1], color=color, ls=ls, lw=1.7, label=key)
        ax.axvline(forward, color='gray', ls=':', alpha=.6, label='x = 20m cut')
        ax.scatter([0], [0], marker='^', s=70, c='black')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=.2)
        ax.set_title(f"{original['id']} | {NAVS[nav[i]]}\n"
                     f"full delta={full_delta[i]:+.3f}m")
        ax.set_xlabel('Ego x (m)')
        ax.set_ylabel('Ego y (m)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, .12, 1, 1))
    fig.savefig(out / 'example_paths.png', dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='V4 paired BEV forward range sensitivity, no retraining')
    ap.add_argument('--data', default='data/synthetic_bev_paths_v4.jsonl')
    ap.add_argument('--markings-checkpoint', default='outputs/ablation_v4/reference_markings.pt')
    ap.add_argument('--full-checkpoint', default='checkpoints/best_path_v4.pt')
    ap.add_argument('--forward-m', type=float, default=20.)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--output-dir', default='outputs/range_sensitivity_20m')
    args = ap.parse_args()
    if args.forward_m <= 0:
        ap.error('--forward-m must be > 0')
    for path in (args.data, args.markings_checkpoint, args.full_checkpoint):
        if not Path(path).is_file():
            ap.error('File not found: ' + path)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    full_ds = BEVDataset(args.data, split='val', seed=args.seed, augment=False)
    cropped_ds = ClippedDataset(args.data, split='val', seed=args.seed,
                                augment=False, forward_m=args.forward_m)
    assert len(full_ds) == len(cropped_ds)
    assert all(a['id'] == b['id'] for a, b in zip(full_ds.rows, cropped_ds.rows))
    print(f'Scenes={len(full_ds)}; BEV clipping: ego x <= {args.forward_m:g}m; rear BEV retained.', flush=True)
    print('Reference, navigation, GT unchanged. Original and 20m predictions evaluated on identical scenes.', flush=True)

    preds, errors = {}, {}
    nav = np.array([{'straight': 0, 'left': 1, 'right': 2}[r['navigation']] for r in full_ds.rows])
    gt = np.array([r['gt_path'] for r in full_ds.rows], dtype=np.float32)
    for mode, ckpt in (('reference_markings', args.markings_checkpoint), ('full', args.full_checkpoint)):
        for condition, dataset in (('original', full_ds), ('20m', cropped_ds)):
            key = mode + '_' + condition
            preds[key] = evaluate_model(ckpt, mode, dataset, args)
            errors[key] = np.linalg.norm(preds[key] - gt, axis=-1)
            score = summarize(errors[key], np.arange(len(nav)))
            print(f'{key:30s} mean={score["path_mean_m"]:.3f}m '
                  f'end={score["endpoint_m"]:.3f}m p95={score["p95_scene_mean_m"]:.3f}m', flush=True)
    metrics = []
    for key, err in errors.items():
        for name, ids in (('all', np.arange(len(nav))), ('straight', np.where(nav==0)[0]),
                          ('left', np.where(nav==1)[0]), ('right', np.where(nav==2)[0]),
                          ('turns', np.where(nav!=0)[0])):
            score = summarize(err, ids)
            if score is not None:
                metrics.append({'variant': key, 'group': name, **score})
    with (out/'metrics.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    with (out/'per_scene.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(['scene_id', 'navigation', *[k+'_mean_m' for k in errors],
                         *[k+'_end_m' for k in errors]])
        for i, r in enumerate(full_ds.rows):
            writer.writerow([r['id'], r['navigation'], *[errors[k][i].mean() for k in errors],
                             *[errors[k][i,-1] for k in errors]])
    with (out/'metrics.json').open('w', encoding='utf8') as f:
        json.dump({'config': {'forward_m': args.forward_m, 'clip_rule': 'x <= forward_m; retain x < 0'},
                   'rows': metrics}, f, indent=2)
    plot_curves(errors, nav, out)
    plot_scenes(full_ds, cropped_ds, preds, errors, nav, args.forward_m, out)
    print('\nPaired impact (20m minus original; positive means worse):')
    for mode in MODES:
        delta = errors[mode+'_20m'].mean(axis=1)-errors[mode+'_original'].mean(axis=1)
        for name, ids in (('all', np.arange(len(nav))), ('turns', np.where(nav!=0)[0])):
            print(f'  {mode:20s} {name:7s}: {delta[ids].mean():+.3f} m')
    print('Saved:', out/'metrics.csv', out/'per_scene.csv', out/'error_vs_distance.png', out/'example_paths.png')


if __name__ == '__main__':
    main()
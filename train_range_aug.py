"""BEV Path V4 range augmentation: same model, same labels, visibility varies.

Train: python -u train_range_aug.py train --mode all --data data/synthetic_bev_paths_v4.jsonl
Compare: python -u train_range_aug.py evaluate --data data/synthetic_bev_paths_v4.jsonl

Run from existing bev_trajectory_demo root. No original checkpoints are modified.
"""
import argparse
import copy
import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from datasets import BEVDataset
from models import TrajectoryModel
from train import path_loss

MODES = ('reference_markings', 'full')
RANGES = (None, 60., 40., 20., 10.)
PROBS = (0.30, 0.20, 0.20, 0.20, 0.10)
NAVS = ('straight', 'left', 'right')


def clip_polyline_xmax(points, xmax, eps=1e-7):
    """Clip polyline to x<=xmax without joining invisible gaps; rear is retained."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return []
    fragments, current = [], []

    def append(point):
        if not current or np.linalg.norm(point - current[-1]) > eps:
            current.append(point.copy())

    def finish():
        if len(current) >= 2 and np.linalg.norm(np.diff(current, axis=0), axis=1).sum() > 1e-3:
            fragments.append(np.asarray(current).round(5).tolist())
        current.clear()

    for a, b in zip(pts[:-1], pts[1:]):
        ain, bin_ = a[0] <= xmax + eps, b[0] <= xmax + eps
        if ain and bin_:
            append(a)
            append(b)
        elif ain and not bin_:
            append(a)
            t = (xmax - a[0]) / (b[0] - a[0])
            append(a + t*(b-a))
            finish()
        elif not ain and bin_:
            t = (xmax - a[0]) / (b[0] - a[0])
            append(a + t*(b-a))
            append(b)
        else:
            finish()
    finish()
    return fragments


def clip_elements(elements, xmax):
    out = []
    for element in elements:
        for points in clip_polyline_xmax(element['points'], xmax):
            item = dict(element)
            item['points'] = points
            out.append(item)
    return out


def version_at_range(base, xmax):
    """Copy metadata; preserve exact same row order, reference, navigation and GT."""
    if xmax is None:
        return base
    ds = copy.copy(base)
    ds.rows = []
    for original in base.rows:
        row = dict(original)
        row['lane_markings'] = clip_elements(original['lane_markings'], xmax)
        row['road_boundaries'] = clip_elements(original['road_boundaries'], xmax)
        ds.rows.append(row)
    return ds


class RangeMix(Dataset):
    """Choose random range for each retrieval, changing only the BEV geometry."""
    def __init__(self, base, ranges=RANGES, probs=PROBS):
        self.variants = [version_at_range(base, r) for r in ranges]
        self.probs = probs
        self.path_points = base.path_points
        self.spacing = base.spacing

    def __len__(self):
        return len(self.variants[0])

    def __getitem__(self, index):
        variant = random.choices(range(len(self.variants)), weights=self.probs, k=1)[0]
        return self.variants[variant][index]


def mask_inputs(batch, mode):
    if mode == 'full':
        return batch
    valid = batch['line_valid'].clone()
    is_marking = batch['lines'][:, :, :, 5].amax(-1) > 0.5
    valid &= is_marking
    result = dict(batch)
    result['line_valid'] = valid
    result['lines'] = batch['lines'] * valid[:, :, None, None].to(batch['lines'].dtype)
    return result


def prepare(batch, mode, device):
    return mask_inputs({k: v.to(device) for k, v in batch.items()}, mode)


def predict_errors(model, dataset, mode, args):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    errors = []
    model.eval()
    with torch.inference_mode():
        for raw in loader:
            batch = prepare(raw, mode, args.device)
            prediction = model(batch)
            errors.append(torch.linalg.vector_norm(prediction - batch['target_xy'], dim=-1).cpu().numpy())
    return np.concatenate(errors, axis=0)


def summarize(error):
    return dict(count=int(len(error)), mean_m=float(error.mean()),
                end_m=float(error[:, -1].mean()),
                p95_scene_m=float(np.percentile(error.mean(axis=1), 95)))


def load_model(path, expected_mode, args, horizon):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('model_type') != 'arc_length_path_v4':
        raise ValueError('Not a V4 path checkpoint: ' + str(path))
    if state.get('ablation_mode', 'full') != expected_mode:
        raise ValueError('Mode mismatch: ' + str(path))
    if state['horizon'] != horizon:
        raise ValueError('Horizon mismatch: ' + str(path))
    model = TrajectoryModel(horizon=horizon).to(args.device)
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    return model


def train_one(args, mode):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    base_train = BEVDataset(args.data, 'train', args.seed, augment=True)
    base_val = BEVDataset(args.data, 'val', args.seed, augment=False)
    train_ds = RangeMix(base_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers)
    val_ds = {label: version_at_range(base_val, rng) for label, rng in
              [('original', None), ('20m', 20.)]}
    model = TrajectoryModel(horizon=base_train.path_points).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = float('inf')
    target = Path(args.output_dir) / f'range_aug_{mode}.pt'
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f'[{mode}] train={len(base_train)} val={len(base_val)}; ranges={RANGES}; probabilities={PROBS}', flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.
        for raw in train_loader:
            batch = prepare(raw, mode, args.device)
            pred = model(batch)
            loss, _, _ = path_loss(pred, batch['target_xy'], base_train.spacing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            running += loss.item() * pred.shape[0]
        e_original = predict_errors(model, val_ds['original'], mode, args)
        e_20 = predict_errors(model, val_ds['20m'], mode, args)
        orig, short = summarize(e_original), summarize(e_20)
        # Balance normal FOV precision and short-FOV robustness; choose once, before testing.
        selection = (orig['mean_m'] + short['mean_m']) / 2
        print(f'{mode:20s} epoch {epoch:03d} loss={running/len(train_ds):.4f} '
              f'original={orig["mean_m"]:.3f}m 20m={short["mean_m"]:.3f}m '
              f'selection={selection:.3f}m', flush=True)
        if selection < best:
            best = selection
            torch.save(dict(model=model.state_dict(), horizon=base_train.path_points,
                            spacing=base_train.spacing, epoch=epoch, seed=args.seed,
                            model_type='arc_length_path_v4', ablation_mode=mode,
                            range_aug=True, range_choices=['original',60,40,20,10],
                            range_probabilities=list(PROBS), best_selection_score=best), target)
    print('Saved best checkpoint:', target, flush=True)


def evaluate(args):
    base = BEVDataset(args.data, 'val', args.seed, augment=False)
    variants = {'original': base, '20m': version_at_range(base, 20.)}
    weights = {
        'reference_markings_baseline': (args.markings_checkpoint, 'reference_markings'),
        'full_baseline': (args.full_checkpoint, 'full'),
        'reference_markings_range_aug': (str(Path(args.output_dir) / 'range_aug_reference_markings.pt'), 'reference_markings'),
        'full_range_aug': (str(Path(args.output_dir) / 'range_aug_full.pt'), 'full'),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    errors, rows = {}, []
    for name, (checkpoint, mode) in weights.items():
        if not Path(checkpoint).is_file():
            raise FileNotFoundError('Missing checkpoint: ' + checkpoint)
        model = load_model(checkpoint, mode, args, base.path_points)
        for regime, dataset in variants.items():
            e = predict_errors(model, dataset, mode, args)
            errors[f'{name}_{regime}'] = e
            nav = np.array([{'straight':0,'left':1,'right':2}[r['navigation']] for r in base.rows])
            for group, idx in [('all', np.ones(len(nav), dtype=bool)),
                               ('straight', nav==0), ('left', nav==1), ('right', nav==2), ('turns', nav!=0)]:
                stats = summarize(e[idx])
                rows.append(dict(model=name, fov=regime, group=group, **stats))
            total = summarize(e)
            print(f'{name:31s} {regime:8s} path_mean={total["mean_m"]:.3f}m '
                  f'endpoint={total["end_m"]:.3f}m p95={total["p95_scene_m"]:.3f}m', flush=True)
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (output / 'comparison.json').open('w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    nav = np.array([{'straight':0,'left':1,'right':2}[r['navigation']] for r in base.rows])
    x = np.arange(base.path_points) * base.spacing
    for ax, (group, idx) in zip(axes, [('all', np.ones(len(nav), dtype=bool)), ('turns', nav!=0)]):
        for name in weights:
            for fov, ls in [('original','-'), ('20m','--')]:
                e = errors[f'{name}_{fov}'][idx]
                ax.plot(x, e.mean(axis=0), ls=ls, lw=1.6, label=f'{name} / {fov}')
        ax.set(title=group, xlabel='Path arc length (m)')
        ax.grid(alpha=.25)
    axes[0].set_ylabel('Mean position error (m)')
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / 'error_vs_distance.png', dpi=160)
    plt.close(fig)
    print('Saved:', output / 'comparison.csv', output / 'error_vs_distance.png', flush=True)


def main():
    parser = argparse.ArgumentParser(description='V4 range-augmented path training and evaluation')
    sub = parser.add_subparsers(dest='command', required=True)
    for action in ('train', 'evaluate'):
        p = sub.add_parser(action)
        p.add_argument('--data', default='data/synthetic_bev_paths_v4.jsonl')
        p.add_argument('--seed', type=int, default=42)
        p.add_argument('--batch-size', type=int, default=32 if action=='train' else 64)
        p.add_argument('--workers', type=int, default=0)
        p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
        p.add_argument('--output-dir', default='outputs/range_aug_v4')
        if action == 'train':
            p.add_argument('--mode', choices=('all',)+MODES, default='all')
            p.add_argument('--epochs', type=int, default=60)
            p.add_argument('--lr', type=float, default=3e-4)
        else:
            p.add_argument('--markings-checkpoint', default='outputs/ablation_v4/reference_markings.pt')
            p.add_argument('--full-checkpoint', default='checkpoints/best_path_v4.pt')
    args = parser.parse_args()
    if not Path(args.data).is_file():
        parser.error('Missing dataset: ' + args.data)
    if args.command == 'train':
        for mode in MODES if args.mode == 'all' else (args.mode,):
            train_one(args, mode)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
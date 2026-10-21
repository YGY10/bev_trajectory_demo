"""Controlled A/B/C input ablation for the V4 time-free BEV path model.

Run from the bev_trajectory_demo project root after copying this file there.
Uses existing datasets/, models/, and train.path_loss; no edits required.
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import BEVDataset
from models import TrajectoryModel
from train import path_loss

MODES = ('reference_only', 'reference_markings', 'full')
NAVS = ('straight', 'left', 'right')


def mask_inputs(batch, mode):
    """Retain all tokens for shape compatibility; mask out excluded geometry tokens."""
    if mode not in MODES:
        raise ValueError(mode)
    if mode == 'full':
        return batch
    batch = dict(batch)
    valid = batch['line_valid'].clone()
    if mode == 'reference_only':
        valid.zero_()
    elif mode == 'reference_markings':
        is_marking = batch['lines'][:, :, :, 5].amax(dim=-1) > 0.5
        valid &= is_marking
    batch['line_valid'] = valid
    # Prevent geometry from excluded elements from entering any computations.
    batch['lines'] = batch['lines'] * valid[:, :, None, None].to(batch['lines'].dtype)
    return batch


def loader(path, split, seed, batch_size, workers=0):
    ds = BEVDataset(path, split=split, seed=seed, augment=(split == 'train'))
    return ds, DataLoader(ds, batch_size=batch_size, shuffle=(split == 'train'), num_workers=workers)


def train_one(args, mode):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    ds_train, dl_train = loader(args.data, 'train', args.seed, args.batch_size, args.workers)
    ds_val, dl_val = loader(args.data, 'val', args.seed, args.batch_size, args.workers)
    model = TrajectoryModel(horizon=ds_train.path_points).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = float('inf')
    output = Path(args.output_dir) / f'{mode}.pt'
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.
        for batch in dl_train:
            batch = mask_inputs({k: v.to(args.device) for k, v in batch.items()}, mode)
            pred = model(batch)
            loss, _, _ = path_loss(pred, batch['target_xy'], ds_train.spacing)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total += loss.item() * pred.shape[0]
        model.eval()
        n, error_sum, endpoint_sum = 0, 0., 0.
        with torch.inference_mode():
            for batch in dl_val:
                batch = mask_inputs({k: v.to(args.device) for k, v in batch.items()}, mode)
                pred = model(batch)
                dist = torch.linalg.vector_norm(pred - batch['target_xy'], dim=-1)
                n += len(dist)
                error_sum += dist.mean(dim=1).sum().item()
                endpoint_sum += dist[:, -1].sum().item()
        mean = error_sum / n
        end = endpoint_sum / n
        print(f'{mode:20s} epoch {epoch:03d} train_loss={total / len(ds_train):.4f} '
              f'val_path_mean={mean:.3f}m val_path_end={end:.3f}m', flush=True)
        if mean < best:
            best = mean
            torch.save({'model': model.state_dict(), 'horizon': ds_train.path_points,
                        'spacing': ds_train.spacing, 'epoch': epoch,
                        'seed': args.seed, 'best_val_path_mean': mean,
                        'model_type': 'arc_length_path_v4', 'ablation_mode': mode}, output)
    print('Best model saved:', output)


def score_one(args, mode, checkpoint):
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if ckpt.get('model_type') != 'arc_length_path_v4':
        raise ValueError(f'Not a V4 checkpoint: {checkpoint}')
    if ckpt.get('ablation_mode', 'full') != mode:
        raise ValueError(f'Mode mismatch: checkpoint={checkpoint} requested={mode}')
    if ckpt['seed'] != args.seed:
        raise ValueError(f'Seed mismatch: checkpoint uses {ckpt["seed"]}, evaluation requests {args.seed}')
    dataset, dl = loader(args.data, 'val', args.seed, args.batch_size, args.workers)
    model = TrajectoryModel(horizon=ckpt['horizon']).to(args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    all_errors, all_nav = [], []
    with torch.inference_mode():
        for batch in dl:
            batch = mask_inputs({k: v.to(args.device) for k, v in batch.items()}, mode)
            pred = model(batch)
            errors = torch.linalg.vector_norm(pred - batch['target_xy'], dim=-1)
            all_errors.append(errors.cpu().numpy())
            all_nav.append(batch['navigation'].cpu().numpy())
    errors = np.concatenate(all_errors, axis=0)
    navs = np.concatenate(all_nav, axis=0)
    rows = []
    for nav in (-1, 0, 1, 2):
        sub = errors if nav == -1 else errors[navs == nav]
        if len(sub) == 0:
            continue
        rows.append({'mode': mode, 'group': 'all' if nav == -1 else NAVS[nav],
                     'count': len(sub), 'path_mean_m': float(sub.mean()),
                     'path_end_m': float(sub[:, -1].mean()),
                     'path_p95_sample_mean_m': float(np.percentile(sub.mean(axis=1), 95))})
    return rows, errors


def evaluate(args):
    paths = {
        'reference_only': Path(args.output_dir) / 'reference_only.pt',
        'reference_markings': Path(args.output_dir) / 'reference_markings.pt',
        'full': Path(args.full_checkpoint),
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing checkpoints: ' + ', '.join(missing))
    all_rows = []
    curves = {}
    for mode, path in paths.items():
        rows, errors = score_one(args, mode, path)
        all_rows.extend(rows)
        curves[mode] = errors.mean(axis=0)
        overall = rows[0]
        print(f'{mode:20s} n={overall["count"]} path_mean={overall["path_mean_m"]:.3f}m '
              f'endpoint={overall["path_end_m"]:.3f}m', flush=True)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'comparison.csv').open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    with (out / 'comparison.json').open('w', encoding='utf-8') as fp:
        json.dump(all_rows, fp, indent=2, ensure_ascii=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    for mode, y in curves.items():
        ax.plot(np.arange(len(y)), y, label=mode)
    ax.set(xlabel='Arc length index (m)', ylabel='Mean position error (m)',
           title='Held-out V4 path error vs distance')
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / 'comparison.png', dpi=180)
    plt.close(fig)
    print('Saved:', out / 'comparison.csv', out / 'comparison.png')
    print('Note: this experiment uses fixed V4 dataset and identical validation partition.')
    print('Full model checkpoint may have a different training RNG trajectory; repeat seeds for rigorous comparisons.')


def main():
    ap = argparse.ArgumentParser(description='A/B/C input ablation on V4 80m paths')
    ap.add_argument('command', choices=('train', 'evaluate'))
    ap.add_argument('--mode', choices=MODES + ('all',), default='all')
    ap.add_argument('--data', default='data/synthetic_bev_paths_v4.jsonl')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--output-dir', default='outputs/ablation_v4')
    ap.add_argument('--full-checkpoint', default='checkpoints/best_path_v4.pt')
    args = ap.parse_args()
    if args.command == 'train':
        modes = ('reference_only', 'reference_markings') if args.mode == 'all' else (args.mode,)
        for mode in modes:
            train_one(args, mode)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
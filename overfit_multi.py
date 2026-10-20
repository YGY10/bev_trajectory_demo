"""Memorize 10 fixed synthetic scenes; a diagnostic, NOT a validation benchmark.

Run: python overfit_multi.py --split train --steps 3000 --lr 3e-4
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from datasets import BEVDataset
from models import TrajectoryModel

NAV_NAMES = {0: 'straight', 1: 'left', 2: 'right'}


def select_balanced(ds, counts):
    chosen = []
    for nav, required in [(1, counts[0]), (2, counts[1]), (0, counts[2])]:
        found = [i for i, r in enumerate(ds.rows) if r['navigation'] == NAV_NAMES[nav]]
        if len(found) < required:
            raise ValueError(f'Need {required} {NAV_NAMES[nav]} scenes, found {len(found)}')
        chosen += found[:required]
    return chosen


def evaluate(model, batch):
    model.eval()
    with torch.no_grad():
        prediction = model(batch)
        distance = torch.linalg.vector_norm(prediction - batch['target_xy'], dim=-1)
        ade = distance.mean(-1).cpu().numpy()
        fde = distance[:, -1].cpu().numpy()
    return prediction.detach().cpu().numpy(), ade, fde


def plot_examples(samples, prediction, indices, out_path, ade, fde):
    fig, axes = plt.subplots(2, 5, figsize=(20, 9), constrained_layout=True)
    for ax, sample, pred, dataset_index, a, f in zip(
        axes.flat, samples, prediction, indices, ade, fde
    ):
        lines = sample['lines'].numpy()
        valid = sample['line_valid'].numpy()
        for line in lines[valid]:
            is_boundary = line[0, 6] > 0.5
            ax.plot(line[:, 0], line[:, 1], color='#555555' if is_boundary else '#c6ae53', alpha=.55, lw=1)
        ref = sample['reference'].numpy()
        gt = sample['target_xy'].numpy()
        ax.plot(ref[:, 0], ref[:, 1], color='orange', ls='--', lw=1, label='Reference')
        ax.plot(gt[:, 0], gt[:, 1], color='green', lw=2, label='GT')
        ax.plot(pred[:, 0], pred[:, 1], color='red', lw=2, label='Prediction')
        ax.scatter([0], [0], color='blue', marker='^', s=35)
        nav = NAV_NAMES[int(sample['navigation'])]
        ax.set_title(f'{nav} | idx={dataset_index}\nADE={a:.2f}m FDE={f:.2f}m', fontsize=10)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Ten-scene memorization test (training examples, not validation)', fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/synthetic_bev_trajectories.jsonl')
    parser.add_argument('--split', choices=['train', 'val', 'all'], default='train')
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-dir', default='outputs/overfit_multi')
    args = parser.parse_args()
    if args.steps < 1 or args.log_every < 1:
        parser.error('--steps and --log-every must be >= 1')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # No training augmentation: fixed samples are required to test memorization.
    ds = BEVDataset(args.data, split=args.split, seed=args.seed, augment=False)
    indices = select_balanced(ds, counts=(4, 3, 3))
    samples = [ds[i] for i in indices]
    batch = {k: v.to(args.device) for k, v in default_collate(samples).items()}
    model = TrajectoryModel(horizon=ds.horizon).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    dest = Path(args.output_dir)
    dest.mkdir(parents=True, exist_ok=True)

    print(f'device={args.device}, split={args.split}, selected={indices}, horizon={ds.horizon}', flush=True)
    print('navigation:', [ds.rows[i]['navigation'] for i in indices], flush=True)
    history = []
    best_ade = float('inf')
    best_step = -1
    best_prediction = None
    best_individual_ade = None
    best_individual_fde = None
    for step in range(1, args.steps + 1):
        model.train()
        pred = model(batch)
        loss = torch.nn.functional.smooth_l1_loss(pred, batch['target_xy'], beta=1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            prediction, ade, fde = evaluate(model, batch)
            mean_ade = float(np.mean(ade))
            mean_fde = float(np.mean(fde))
            record = {'step': step, 'train_loss': float(loss.item()), 'ADE': mean_ade, 'FDE': mean_fde,
                      'max_ADE': float(np.max(ade)), 'max_FDE': float(np.max(fde))}
            for category in ('left', 'right', 'straight'):
                category_mask = np.array([ds.rows[i]['navigation'] == category for i in indices])
                record[f'{category}_ADE'] = float(np.mean(ade[category_mask]))
                record[f'{category}_FDE'] = float(np.mean(fde[category_mask]))
            history.append(record)
            print(f'Step {step:04d} | loss={loss.item():.5f} | ADE={mean_ade:.3f}m | '
                  f'FDE={mean_fde:.3f}m | max_ADE={max(ade):.3f}m | '
                  f'L/R/S ADE={record["left_ADE"]:.2f}/{record["right_ADE"]:.2f}/{record["straight_ADE"]:.2f}m',
                  flush=True)
            if mean_ade < best_ade:
                best_ade = mean_ade
                best_step = step
                best_prediction = prediction.copy()
                best_individual_ade = ade.copy()
                best_individual_fde = fde.copy()
                torch.save({'model': model.state_dict(), 'horizon': ds.horizon, 'step': step,
                            'indices': indices, 'seed': args.seed, 'best_train_ade': best_ade},
                           dest / 'best.pt')

    with (dest / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    plot_examples(samples, best_prediction, indices, dest / 'trajectories.png',
                  best_individual_ade, best_individual_fde)
    with (dest / 'per_scene.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset_index', 'scene_id', 'navigation', 'ADE_m', 'FDE_m'])
        for idx, a, fe in zip(indices, best_individual_ade, best_individual_fde):
            writer.writerow([idx, ds.rows[idx]['id'], ds.rows[idx]['navigation'], float(a), float(fe)])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([r['step'] for r in history], [r['ADE'] for r in history], label='ADE')
    ax.plot([r['step'] for r in history], [r['FDE'] for r in history], label='FDE')
    ax.set(xlabel='Optimizer step', ylabel='Error (m)', title='Ten-scene memorization diagnostic')
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(dest / 'learning_curve.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Best checkpoint step={best_step}, ADE={best_ade:.3f}m; saved outputs to {dest}', flush=True)
    if np.max(best_individual_ade) < 0.5 and np.max(best_individual_fde) < 0.5:
        print('PASS: all 10 examples have ADE and FDE below 0.5 m (memorization only).', flush=True)
    else:
        print('NOT YET PASS: at least one example has ADE or FDE >= 0.5 m.', flush=True)


if __name__ == '__main__':
    main()

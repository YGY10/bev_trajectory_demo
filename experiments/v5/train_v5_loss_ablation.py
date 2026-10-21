#!/usr/bin/env python3
"""V5 controlled ablation: original path loss vs weak reference residual penalty.
Does not modify existing V4/V5 scripts or overwrite their checkpoints.
"""
import argparse
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import BEVDataset
from models import TrajectoryModel
from train import path_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-data', default='data/v5_splits/train.jsonl')
    p.add_argument('--val-data', default='data/v5_splits/val.jsonl')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-lines', type=int, default=32)
    p.add_argument('--lambda-residual', type=float, default=0.01,
                   help='Weight on mean residual magnitude in meters (0=original loss)')
    p.add_argument('--output', default='checkpoints/best_path_v5_residual001.pt')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--workers', type=int, default=0)
    args = p.parse_args()
    if args.lambda_residual < 0:
        p.error('--lambda-residual must be nonnegative')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train = BEVDataset(args.train_data, split='all', augment=False, max_lines=args.max_lines)
    train.augment = True  # split=all disables this by default; no dataset re-splitting.
    val = BEVDataset(args.val_data, split='all', augment=False, max_lines=args.max_lines)
    if train.path_points != val.path_points or train.spacing != val.spacing:
        raise ValueError('Train/val GT horizon or spacing mismatch')
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = TrajectoryModel(horizon=train.path_points).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = float('inf')
    if Path(args.output).exists():
        raise FileExistsError(f'Refusing to overwrite existing checkpoint: {args.output}')
    print(f'lambda_residual={args.lambda_residual} max_lines={args.max_lines} '
          f'train={len(train)} val={len(val)} device={args.device}', flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = total_regularizer = 0.0
        for batch in train_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            pred = model(batch)
            base_loss, _, _ = path_loss(pred, batch['target_xy'], train.spacing)
            # pred and reference both begin at ego origin; path-level L2 norm.
            residual_magnitude = torch.linalg.vector_norm(
                pred - batch['reference'], dim=-1).mean()
            loss = base_loss + args.lambda_residual * residual_magnitude
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * pred.shape[0]
            total_regularizer += float(residual_magnitude.item()) * pred.shape[0]
        model.eval()
        all_d, all_spacing = [], []
        with torch.inference_mode():
            for batch in val_loader:
                batch = {k: v.to(args.device) for k, v in batch.items()}
                pred = model(batch)
                all_d.append(torch.linalg.vector_norm(pred - batch['target_xy'], dim=-1).cpu())
                all_spacing.append((torch.linalg.vector_norm(pred[:, 1:] - pred[:, :-1], dim=-1)
                                    - train.spacing).abs().cpu())
        dist = torch.cat(all_d)
        mean = float(dist.mean())
        endpoint = float(dist[:, -1].mean())
        spacing = float(torch.cat(all_spacing).mean())
        print(f'Epoch {epoch:03d}: train_loss={total/len(train):.4f} '
              f'train_residual_m={total_regularizer/len(train):.4f} '
              f'val_path_mean={mean:.3f}m val_path_end={endpoint:.3f}m '
              f'val_spacing_err={spacing:.3f}m', flush=True)
        if mean < best:
            best = mean
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model': model.state_dict(), 'model_type': 'arc_length_path_v5',
                        'horizon': train.path_points, 'spacing': train.spacing,
                        'epoch': epoch, 'seed': args.seed, 'best_val_path_mean': best,
                        'lambda_residual': args.lambda_residual, 'max_lines': args.max_lines}, args.output)
    print('Saved best checkpoint:', args.output)


if __name__ == '__main__':
    main()

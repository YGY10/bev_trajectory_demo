"""Train 81-point arc-length BEV path model on synthetic V4 dataset."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import BEVDataset
from models import TrajectoryModel


def path_loss(pred, target, spacing):
    # Compare geometric samples at equal desired arc lengths; no time/velocity.
    fit = nn.functional.smooth_l1_loss(pred, target, beta=1.0)
    dp = pred[:, 1:] - pred[:, :-1]
    segment_length = torch.linalg.vector_norm(dp, dim=-1)
    equal_arc = (segment_length - spacing).abs().mean()
    # Weak second-difference penalty avoids sharp zigzags without flattening turns.
    bend = (pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]).square().sum(-1).mean()
    loss = fit + 0.3 * equal_arc + 0.02 * bend
    return loss, fit.detach(), equal_arc.detach()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/synthetic_bev_paths_v4.jsonl')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', default='checkpoints/best_path_v4.pt')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--workers', type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train = BEVDataset(args.data, 'train', args.seed, augment=True)
    val = BEVDataset(args.data, 'val', args.seed, augment=False)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=args.workers)
    model = TrajectoryModel(horizon=train.path_points).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            pred = model(batch)
            loss, _, _ = path_loss(pred, batch['target_xy'], train.spacing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * pred.shape[0]
        model.eval()
        all_d, all_spacing = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(args.device) for k, v in batch.items()}
                pred = model(batch)
                d = torch.linalg.vector_norm(pred - batch['target_xy'], dim=-1)
                seg = torch.linalg.vector_norm(pred[:, 1:] - pred[:, :-1], dim=-1)
                all_d.append(d.cpu())
                all_spacing.append((seg - train.spacing).abs().cpu())
        dist = torch.cat(all_d)
        spacing_err = torch.cat(all_spacing).mean().item()
        mean_err, end_err = dist.mean().item(), dist[:, -1].mean().item()
        print(f'Epoch {epoch:03d}: train_loss={total / len(train):.4f} '
              f'val_path_mean={mean_err:.3f}m val_path_end={end_err:.3f}m '
              f'val_spacing_err={spacing_err:.3f}m', flush=True)
        if mean_err < best:
            best = mean_err
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model': model.state_dict(), 'horizon': train.path_points,
                        'spacing': train.spacing, 'epoch': epoch,
                        'seed': args.seed, 'best_val_path_mean': best,
                        'model_type': 'arc_length_path_v4'}, args.output)
    print('Saved best checkpoint:', args.output)


if __name__ == '__main__':
    main()

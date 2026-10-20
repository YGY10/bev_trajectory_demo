"""Single-scene memorization diagnostic (not generalization evaluation)."""
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import BEVDataset
from models import TrajectoryModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/synthetic_bev_trajectories.jsonl')
    p.add_argument('--split', choices=['train', 'val', 'all'], default='val')
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--out', default='outputs/overfit_one.png')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    ds = BEVDataset(args.data, split=args.split, seed=42, augment=False)
    if not 0 <= args.index < len(ds):
        raise ValueError(f'Index {args.index} not in [0, {len(ds)-1}]')
    example = ds[args.index]
    batch = {k: v.unsqueeze(0).to(args.device) for k,v in example.items()}
    target = batch['target_xy']
    model = TrajectoryModel(horizon=ds.horizon).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    print(f'Split={args.split}, index={args.index}, device={args.device}, horizon={ds.horizon}', flush=True)
    for step in range(1, args.steps + 1):
        model.train()
        pred = model(batch)
        loss = torch.nn.functional.smooth_l1_loss(pred, target, beta=1.0)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                pred_eval = model(batch)
                dist = torch.linalg.vector_norm(pred_eval-target, dim=-1)
                ade = dist.mean().item()
                fde = dist[:, -1].mean().item()
            print(f'Step {step:04d} | loss={loss.item():.5f} | ADE={ade:.4f}m | FDE={fde:.4f}m', flush=True)

    model.eval()
    with torch.no_grad():
        prediction = model(batch)[0].cpu().numpy()
    gt = target[0].cpu().numpy()
    ref = example['reference'].numpy()
    lines = example['lines'].numpy()
    valid = example['line_valid'].numpy()
    fig, ax = plt.subplots(figsize=(8, 8))
    for line in lines[valid]:
        is_boundary = line[0,6] > 0.5
        ax.plot(line[:,0], line[:,1], color='#444444' if is_boundary else '#bdad56', lw=1.5, alpha=0.7)
    ax.plot(ref[:,0], ref[:,1], color='orange', ls='--', label='Biased reference')
    ax.plot(gt[:,0], gt[:,1], color='green', lw=2.5, label='GT')
    ax.plot(prediction[:,0], prediction[:,1], color='red', lw=2.5, label='Prediction')
    ax.scatter([gt[0,0]], [gt[0,1]], marker='^', color='blue', s=70, label='Ego')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=.25)
    ax.legend()
    ax.set_title(f'Overfit single sample | {args.split}[{args.index}] | {args.steps} steps')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('Saved:', args.out, flush=True)

if __name__ == '__main__':
    main()

"""Show sensed BEV polylines, biased reference, 80m GT path and prediction."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import BEVDataset
from models import TrajectoryModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/synthetic_bev_paths_v4.jsonl')
    p.add_argument('--checkpoint', default='checkpoints/best_path_v4.pt')
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--output', default='outputs/path_comparison_v4.png')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if ckpt.get('model_type') != 'arc_length_path_v4':
        raise ValueError('This checkpoint is not a V4 path model')
    ds = BEVDataset(args.data, split='val', seed=ckpt['seed'])
    sample = ds[args.index]
    model = TrajectoryModel(horizon=ckpt['horizon']).to(args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    with torch.no_grad():
        batch = {k: v.unsqueeze(0).to(args.device) for k, v in sample.items()}
        pred = model(batch)[0].cpu().numpy()
    row = ds.rows[args.index]
    fig, ax = plt.subplots(figsize=(10, 7))
    for e in row['lane_markings']:
        pts = np.asarray(e['points'])
        if len(pts):
            c = e.get('color', 'gray')
            ax.plot(pts[:, 0], pts[:, 1], color=c if c != 'white' else '#d9bd84', lw=1.5)
    for e in row['road_boundaries']:
        pts = np.asarray(e['points'])
        if len(pts):
            ax.plot(pts[:, 0], pts[:, 1], color='0.35', lw=2)
    ref = sample['reference'].numpy()
    gt = sample['target_xy'].numpy()
    ax.plot(ref[:, 0], ref[:, 1], '--', color='orange', label='Biased reference')
    ax.plot(gt[:, 0], gt[:, 1], color='green', lw=2.5, label='GT path (80m)')
    ax.plot(pred[:, 0], pred[:, 1], color='red', lw=2.5, label='Predicted path')
    ax.scatter([0], [0], marker='^', s=90, c='blue', label='Ego')
    err = np.linalg.norm(pred - gt, axis=-1)
    ax.set_title(f"{row['id']} / {row['navigation']} | path mean={err.mean():.2f}m | end={err[-1]:.2f}m")
    ax.set_xlabel('ego x (m)'); ax.set_ylabel('ego y (m)')
    ax.axis('equal'); ax.grid(alpha=0.2); ax.legend()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print('Saved:', args.output)


if __name__ == '__main__':
    main()

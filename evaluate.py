"""Evaluate held-out 80m geometric paths, without any time-based metrics."""
import argparse
import torch
from torch.utils.data import DataLoader
from datasets import BEVDataset
from models import TrajectoryModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/synthetic_bev_paths_v4.jsonl')
    p.add_argument('--checkpoint', default='checkpoints/best_path_v4.pt')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if ckpt.get('model_type') != 'arc_length_path_v4':
        raise ValueError('This checkpoint is not a V4 path model')
    dataset = BEVDataset(args.data, split='val', seed=ckpt['seed'])
    model = TrajectoryModel(horizon=ckpt['horizon']).to(args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    dists, spacings = [], []
    nav_dists = {0: [], 1: [], 2: []}
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=args.batch_size):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            pred = model(batch)
            distances = torch.linalg.vector_norm(pred - batch['target_xy'], dim=-1).cpu()
            gap = torch.linalg.vector_norm(pred[:, 1:] - pred[:, :-1], dim=-1)
            spacings.append((gap - dataset.spacing).abs().cpu())
            dists.append(distances)
            for nav in (0, 1, 2):
                matching = batch['navigation'].cpu() == nav
                if matching.any():
                    nav_dists[nav].append(distances[matching])
    d = torch.cat(dists)
    print(f'Scenes: {len(dataset)} | Path mean: {d.mean():.3f} m | '
          f'Path endpoint: {d[:, -1].mean():.3f} m | '
          f'Mean spacing error: {torch.cat(spacings).mean():.3f} m')
    for nav, label in [(0, 'straight'), (1, 'left'), (2, 'right')]:
        if nav_dists[nav]:
            sub = torch.cat(nav_dists[nav])
            print(f'{label}: count={len(sub)} mean={sub.mean():.3f}m endpoint={sub[:, -1].mean():.3f}m')
    print('These are geometric path metrics on synthetic held-out data, not human-likeness validation.')


if __name__ == '__main__':
    main()

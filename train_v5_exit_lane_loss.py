#!/usr/bin/env python3
"""Controlled V5 experiment: visible target-exit-lane lateral supervision.

Model inputs remain unchanged. The lane geometry and masks are ONLY training
supervision, reconstructed from the original generator and actual observed
markings. Requires evaluate_exit_lane_v5.py + data/generate_dataset_v5.py.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import BEVDataset
from models import TrajectoryModel
from train import path_loss
from evaluate_exit_lane_v5 import base_geometry, observed_sides, near_visible


class ExitSupervisedDataset(BEVDataset):
    def __init__(self, path, *, seed=20261021, max_lines=32,
                 visibility_gap=2.0, min_visible_points=5, augment=False):
        super().__init__(path, split='all', augment=False, max_lines=max_lines)
        self.augment = bool(augment)
        self.exit_supervision = []
        self.count_eligible = 0
        for row in self.rows:
            b, v, n, width, _ = base_geometry(row, seed)
            sides = observed_sides(row, b, v, n, width, row['metadata']['exit_lane_index'])
            gt = np.asarray(row['gt_path'], dtype=np.float64)
            station = (gt - b) @ v
            mask = ((station >= 3.0) & near_visible(station, sides[0], visibility_gap)
                    & near_visible(station, sides[1], visibility_gap))
            if int(mask.sum()) < min_visible_points:
                mask[:] = False
            else:
                self.count_eligible += 1
                # An independent geometry check: straight-exit GT is centered
                # on reconstructed target exit lane. Never use GT as lane center.
                deviation = np.abs((gt[mask] - b) @ n)
                if deviation.max() > 0.1:
                    raise ValueError(f'{row["id"]}: exit geometry/GT mismatch '
                                     f'({deviation.max():.3f}m). Check generator seed.')
            self.exit_supervision.append((torch.from_numpy(mask.copy()),
                                          torch.tensor(b, dtype=torch.float32),
                                          torch.tensor(n, dtype=torch.float32)))

    def __getitem__(self, index):
        item = super().__getitem__(index)
        mask, b, n = self.exit_supervision[index]
        item['exit_mask'] = mask
        item['exit_anchor'] = b
        item['exit_normal'] = n
        return item


def exit_lateral_loss(pred, batch):
    """Mean absolute centerline lateral deviation in meters over eligible points.

    Computes a globally point-weighted mean; mask/geometry are never fed to
    the network, only used after forward() as differentiable supervision.
    """
    mask = batch['exit_mask']
    lateral = ((pred - batch['exit_anchor'][:, None, :]) *
               batch['exit_normal'][:, None, :]).sum(-1).abs()
    total_points = mask.sum()
    if total_points.item() == 0:
        return pred.sum() * 0.0, 0
    return (lateral * mask.to(lateral.dtype)).sum() / total_points, int(total_points.item())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-data', default='data/v5_splits/train.jsonl')
    p.add_argument('--val-data', default='data/v5_splits/val.jsonl')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--generator-seed', type=int, default=20261021)
    p.add_argument('--max-lines', type=int, default=32)
    p.add_argument('--lambda-lane', type=float, default=0.02)
    p.add_argument('--visibility-gap', type=float, default=2.0)
    p.add_argument('--min-visible-points', type=int, default=5)
    p.add_argument('--output', default='checkpoints/v5_exit_lane_lambda002.pt')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--workers', type=int, default=0)
    a = p.parse_args()
    if a.lambda_lane < 0: p.error('--lambda-lane must be nonnegative')
    if Path(a.output).exists(): raise FileExistsError('Refusing to overwrite: '+a.output)
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)

    train = ExitSupervisedDataset(a.train_data, seed=a.generator_seed,
                                  max_lines=a.max_lines, visibility_gap=a.visibility_gap,
                                  min_visible_points=a.min_visible_points, augment=True)
    val = ExitSupervisedDataset(a.val_data, seed=a.generator_seed,
                                max_lines=a.max_lines, visibility_gap=a.visibility_gap,
                                min_visible_points=a.min_visible_points, augment=False)
    if train.path_points != val.path_points or train.spacing != val.spacing:
        raise ValueError('Train/validation horizon or spacing mismatch')
    tr_loader = DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=a.workers)
    va_loader = DataLoader(val,batch_size=a.batch_size,shuffle=False,num_workers=a.workers)
    model = TrajectoryModel(horizon=train.path_points).to(a.device)
    opt = torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    best = float('inf')
    print(f'lambda_lane={a.lambda_lane} train={len(train)} val={len(val)} '
          f'eligible_train={train.count_eligible} eligible_val={val.count_eligible} '
          f'generator_seed={a.generator_seed} device={a.device}', flush=True)
    for epoch in range(1,a.epochs+1):
        model.train(); summed_loss=0.; lateral_numer=0.; npoints=0
        for batch in tr_loader:
            batch={k:v.to(a.device) for k,v in batch.items()}
            # pass original model inputs only; supervision is not a model feature.
            pred=model({k:batch[k] for k in ('lines','line_valid','reference','navigation')})
            base,_,_=path_loss(pred,batch['target_xy'],train.spacing)
            lane,n=exit_lateral_loss(pred,batch)
            loss=base+a.lambda_lane*lane
            opt.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            summed_loss+=float(loss.item())*pred.shape[0]
            lateral_numer+=float(lane.detach().item())*n; npoints+=n
        model.eval(); total_ade=0.; total_end=0.; total_spacing=0.; total_items=0; lateral_sum=0.; count=0
        with torch.inference_mode():
            for batch in va_loader:
                batch={k:v.to(a.device) for k,v in batch.items()}
                pred=model({k:batch[k] for k in ('lines','line_valid','reference','navigation')})
                d=torch.linalg.vector_norm(pred-batch['target_xy'],dim=-1)
                seg=(torch.linalg.vector_norm(pred[:,1:]-pred[:,:-1],dim=-1)-train.spacing).abs()
                m,n=exit_lateral_loss(pred,batch)
                total_ade+=d.sum().item();total_end+=d[:,-1].sum().item()
                total_spacing+=seg.sum().item();total_items+=pred.shape[0]
                lateral_sum+=m.item()*n;count+=n
        ade=total_ade/(total_items*train.path_points)
        fde=total_end/total_items
        spe=total_spacing/(total_items*(train.path_points-1))
        exit_lat=lateral_sum/max(count,1)
        print(f'Epoch {epoch:03d}: train_loss={summed_loss/len(train):.4f} '
              f'train_visible_exit_lat={lateral_numer/max(npoints,1):.3f}m '
              f'val_path_mean={ade:.3f}m val_path_end={fde:.3f}m '
              f'val_visible_exit_lat={exit_lat:.3f}m val_spacing_err={spe:.3f}m',flush=True)
        # checkpoint selected by original val ADE for direct comparison; then
        # use independent strict exit eval to assess lateral performance.
        if ade<best:
            best=ade
            Path(a.output).parent.mkdir(parents=True,exist_ok=True)
            torch.save({'model':model.state_dict(),'model_type':'arc_length_path_v5',
                        'horizon':train.path_points,'spacing':train.spacing,
                        'epoch':epoch,'seed':a.seed,'max_lines':a.max_lines,
                        'best_val_path_mean':best,'lambda_lane':a.lambda_lane,
                        'generator_seed':a.generator_seed,'training_variant':'visible_exit_lateral'},a.output)
    print('Saved best checkpoint:',a.output,flush=True)

if __name__=='__main__':
    main()

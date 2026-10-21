#!/usr/bin/env python3
"""Diagnose V5 visible-exit supervision, checkpoint metrics, and gradient alignment.
No training; no checkpoint modification. Run from bev_trajectory_demo root.
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from models import TrajectoryModel
from train import path_loss
from train_v5_exit_lane_loss import ExitSupervisedDataset, exit_lateral_loss

INPUT_KEYS = ('lines', 'line_valid', 'reference', 'navigation')


def masked_stats(points, batch):
    """Absolute perpendicular distance to reconstructed exit center, per masked point."""
    normal = batch['exit_normal'][:, None, :]
    anchor = batch['exit_anchor'][:, None, :]
    dev = ((points - anchor) * normal).sum(-1).abs()
    mask = batch['exit_mask']
    n = int(mask.sum())
    return (float(dev[mask].sum().item()), n, float(dev[mask].max().item()) if n else math.nan)


def summarize_checkpoint(path, ds, device, batch_size, output_csv):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model = TrajectoryModel(horizon=int(ckpt.get('horizon', ds.path_points))).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    sums = dict(gt=0., reference=0., prediction=0., total_points=0, eligible=0,
                ade=0., n_paths=0, model_better=0, ref_better=0)
    records = []
    index = 0
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model({k: batch[k] for k in INPUT_KEYS})
            gt = batch['target_xy']
            ref = batch['reference']
            for label, pts in [('gt', gt), ('reference', ref), ('prediction', pred)]:
                s, n, _ = masked_stats(pts, batch)
                sums[label] += s
                if label == 'gt':
                    sums['total_points'] += n
            sums['ade'] += float(torch.linalg.vector_norm(pred - gt, dim=-1).mean(-1).sum().item())
            sums['n_paths'] += len(pred)
            for i in range(len(pred)):
                mask = batch['exit_mask'][i]
                count = int(mask.sum().item())
                if not count:
                    index += 1
                    continue
                b = batch['exit_anchor'][i]
                nvec = batch['exit_normal'][i]
                def lateral(pts):
                    return float((((pts[i][mask] - b) * nvec).sum(-1)).abs().mean().item())
                p_lat, r_lat, g_lat = lateral(pred), lateral(ref), lateral(gt)
                sums['eligible'] += 1
                sums['model_better'] += p_lat < r_lat
                sums['ref_better'] += r_lat < p_lat
                records.append({'index': index, 'scene_id': ds.rows[index].get('id', ''),
                                'navigation': ds.rows[index].get('navigation', ''),
                                'eligible_points': count, 'gt_lat_m': g_lat,
                                'ref_lat_m': r_lat, 'model_lat_m': p_lat,
                                'improvement_m': r_lat-p_lat})
                index += 1
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]) if records else ['index','scene_id'])
        w.writeheader(); w.writerows(records)
    p = max(sums['total_points'], 1)
    print(f'\nCHECKPOINT {path} | epoch={ckpt.get("epoch")}')
    print(f'  ADE={sums["ade"]/max(sums["n_paths"],1):.4f} m; eligible scenes={sums["eligible"]}; eligible points={sums["total_points"]}')
    for k in ['gt','reference','prediction']:
        print(f'  same-mask {k:10s} lateral={sums[k]/p:.5f} m (point weighted)')
    print(f'  model-better scenes={sums["model_better"]}; reference-better scenes={sums["ref_better"]}')
    print(f'  per-scene CSV: {output_csv}')
    return model


def gradient_diagnostic(model, ds, device, num_batches, batch_size, weight):
    model.eval()  # disable dropout: compare two gradients on the same forward graph
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    params = [p for p in model.parameters() if p.requires_grad]
    rows = []
    for j, batch in enumerate(loader):
        if len(rows) >= num_batches:
            break
        if not bool(batch['exit_mask'].any()):
            continue
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model({k:batch[k] for k in INPUT_KEYS})
        base, _, _ = path_loss(pred, batch['target_xy'], ds.spacing)
        lane, n = exit_lateral_loss(pred, batch)
        if n == 0:
            continue
        grads_b = torch.autograd.grad(base, params, retain_graph=True, allow_unused=True)
        grads_l = torch.autograd.grad(lane, params, allow_unused=True)
        nb = nl = dot = 0.
        for gb, gl in zip(grads_b, grads_l):
            if gb is None or gl is None:
                continue
            b = gb.detach().float().reshape(-1)
            l = gl.detach().float().reshape(-1)
            nb += float(torch.dot(b,b)); nl += float(torch.dot(l,l)); dot += float(torch.dot(b,l))
        nb, nl = math.sqrt(nb), math.sqrt(nl)
        cosine = dot/(nb*nl) if nb > 0 and nl > 0 else math.nan
        ratio = weight*nl/nb if nb > 0 else math.nan
        rows.append([j,n,float(base.item()),float(lane.item()),nb,nl,cosine,ratio])
        del pred, grads_b, grads_l
    print(f'\nGRADIENT CHECK (model.eval(), lambda_lane={weight})')
    print('  batch valid_points base_loss lane_loss |g_base| |g_lane| cosine weighted_grad_ratio')
    for x in rows:
        print('  %3d %4d %.5f %.5f %.5f %.5f %+.3f %.5f' % tuple(x))
    if rows:
        a = np.asarray(rows, dtype=float)
        print(f'  mean gradient cosine={np.nanmean(a[:,6]):+.3f}, mean weighted ratio={np.nanmean(a[:,7]):.4f}')
    else:
        print('  No eligible batches were seen. Increase --grad-batches or sample from a known eligible subset.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data', default='data/v5_splits/val.jsonl')
    p.add_argument('--baseline', default='checkpoints/v5_loss_baseline_retrain.pt')
    p.add_argument('--lane-checkpoint', default='checkpoints/v5_exit_lane_lambda002.pt')
    p.add_argument('--generator-seed', type=int, default=20261021)
    p.add_argument('--max-lines', type=int, default=32)
    p.add_argument('--visibility-gap', type=float, default=2.)
    p.add_argument('--min-visible-points', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--grad-batches', type=int, default=5)
    p.add_argument('--lambda-lane', type=float, default=0.02)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output-dir', default='outputs/v5_lane_diagnostic')
    a=p.parse_args()
    ds=ExitSupervisedDataset(a.data, seed=a.generator_seed,max_lines=a.max_lines,
        visibility_gap=a.visibility_gap,min_visible_points=a.min_visible_points,augment=False)
    print(f'Dataset: {len(ds)} scenes; eligible={ds.count_eligible}')
    # Independent mask check against the same target geometry. Mask is defined using GT station,
    # and should not leak into model.forward().
    counts=np.asarray([int(x[0].sum()) for x in ds.exit_supervision])
    valid=counts[counts>0]
    print(f'Mask points per eligible scene: min={valid.min() if len(valid) else 0}, median={np.median(valid) if len(valid) else 0}, max={valid.max() if len(valid) else 0}')
    out=Path(a.output_dir)
    baseline=summarize_checkpoint(a.baseline,ds,a.device,a.batch_size,out/'baseline_per_scene.csv')
    lane_model=summarize_checkpoint(a.lane_checkpoint,ds,a.device,a.batch_size,out/'lane_per_scene.csv')
    gradient_diagnostic(baseline,ds,a.device,a.grad_batches,a.batch_size,a.lambda_lane)
    del lane_model

if __name__=='__main__':
    main()

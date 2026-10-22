#!/usr/bin/env python3
"""V5 residual scaling sweep, with GT-indexed strict visible-exit metrics.

Run from repository root. Uses the same mask and scene aggregation as
 evaluate_exit_lane_v5.py, and never trains / changes a checkpoint.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from datasets import BEVDataset
from models import TrajectoryModel
from evaluate_exit_lane_v5 import calc_one


def save_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(v):
    return f'{v:.3f}' if np.isfinite(v) else 'N/A'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default='data/v5_splits/val.jsonl')
    ap.add_argument('--checkpoint', nargs=2, metavar=('NAME', 'PATH'), action='append', required=True)
    ap.add_argument('--alphas', nargs='+', type=float, default=[0., .25, .5, .75, 1.])
    ap.add_argument('--generator-seed', type=int, default=20261021)
    ap.add_argument('--max-lines', type=int, default=32)
    ap.add_argument('--station-gap-m', type=float, default=1.5)
    ap.add_argument('--min-points', type=int, default=5)
    ap.add_argument('--center-tolerance-m', type=float, default=.5)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--output-dir', default='outputs/v5_residual_scaling')
    args = ap.parse_args()
    if len(set(n for n, _ in args.checkpoint)) != len(args.checkpoint):
        ap.error('Checkpoint NAME must be unique')
    if not all(np.isfinite(x) and 0 <= x <= 1 for x in args.alphas):
        ap.error('alphas must be finite numbers in [0,1]')
    out = Path(args.output_dir)
    ds = BEVDataset(args.data, split='all', augment=False, max_lines=args.max_lines)
    per_scene = []
    summary = []
    for name, ckpt_path in args.checkpoint:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        if int(ckpt['horizon']) != ds.path_points:
            raise ValueError(f'{name}: checkpoint horizon != dataset points')
        model = TrajectoryModel(horizon=int(ckpt['horizon'])).to(args.device)
        model.load_state_dict(ckpt['model'])
        model.eval()
        predictions = []
        # Cache outputs one time per checkpoint; reuse across every alpha.
        with torch.inference_mode():
            for i in range(len(ds)):
                item = ds[i]
                x = {k: item[k].unsqueeze(0).to(args.device) for k in ('lines', 'line_valid', 'reference', 'navigation')}
                pred = model(x)[0].cpu().numpy().astype(np.float64)
                predictions.append(pred)
        for alpha in args.alphas:
            rows = []
            for i, row in enumerate(ds.rows):
                item = ds[i]
                ref = item['reference'].numpy().astype(np.float64)
                gt = item['target_xy'].numpy().astype(np.float64)
                pred = ref + alpha * (predictions[i] - ref)
                path_err = np.linalg.norm(pred - gt, axis=-1)
                record = {
                    'checkpoint': name, 'alpha': alpha,
                    'scene_id': row['id'], 'navigation': row['navigation'],
                    'ego_distance_m': float(row['metadata']['ego_distance_to_junction_m']),
                    'path_ade_m': float(path_err.mean()),
                    'path_fde_m': float(path_err[-1]),
                }
                strict = calc_one(row, ref, gt, pred, args.generator_seed,
                                  args.station_gap_m, args.min_points, args.center_tolerance_m)
                record.update({
                    'exit_eligible': bool(strict['eligible']),
                    'exit_visible_points': int(strict['visible_target_points']),
                })
                if strict['eligible']:
                    record.update({
                        'ref_exit_lat_m': strict['reference_lat_mean_m'],
                        'pred_exit_lat_m': strict['model_lat_mean_m'],
                        'ref_center_hit': strict['reference_center_within_tol_frac'],
                        'pred_center_hit': strict['model_center_within_tol_frac'],
                        'pred_lane_hit': strict['model_within_lane_frac'],
                        'pred_heading_deg': strict['model_heading_mean_deg'],
                    })
                rows.append(record)
                per_scene.append(record)
            for group, nav in [('all', None), ('straight', 'straight'), ('left', 'left'), ('right', 'right')]:
                selected = [r for r in rows if nav is None or r['navigation'] == nav]
                eligible = [r for r in selected if r['exit_eligible']]
                s = {
                    'checkpoint': name, 'alpha': alpha, 'navigation': group,
                    'total_scenes': len(selected), 'eligible_scenes': len(eligible),
                    'path_ade_m': float(np.mean([r['path_ade_m'] for r in selected])),
                    'path_fde_m': float(np.mean([r['path_fde_m'] for r in selected])),
                    'ref_exit_lat_m': float(np.mean([r['ref_exit_lat_m'] for r in eligible])) if eligible else float('nan'),
                    'pred_exit_lat_m': float(np.mean([r['pred_exit_lat_m'] for r in eligible])) if eligible else float('nan'),
                    'pred_center_hit': float(np.mean([r['pred_center_hit'] for r in eligible])) if eligible else float('nan'),
                    'pred_lane_hit': float(np.mean([r['pred_lane_hit'] for r in eligible])) if eligible else float('nan'),
                    'pred_heading_deg': float(np.mean([r['pred_heading_deg'] for r in eligible])) if eligible else float('nan'),
                }
                s['exit_improvement_vs_ref_m'] = s['ref_exit_lat_m'] - s['pred_exit_lat_m']
                summary.append(s)
                if group == 'all' or group == 'straight':
                    print(f'{name:12s} alpha={alpha:.2f} {group:8s} ADE={s["path_ade_m"]:.3f}m '
                          f'exit={fmt(s["pred_exit_lat_m"])}m '
                          f'ref_exit={fmt(s["ref_exit_lat_m"])}m '
                          f'center_hit={s["pred_center_hit"]*100:.1f}% '
                          f'n={len(eligible)}', flush=True)
    save_csv(out / 'per_scene.csv', per_scene)
    save_csv(out / 'summary.csv', summary)
    print('Saved:', out / 'per_scene.csv', out / 'summary.csv')
    print('Choose alpha only using VAL data; keep TEST held out. Metrics use GT-station masks, same as strict evaluator.')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Compare reference, original V5 prediction, and navigation-scaled prediction.

Run from bev_trajectory_demo repository root. Inference once per scene. Strict
exit masks/metrics exactly reuse evaluate_exit_lane_v5.calc_one. Validation only
until the method is frozen. No changes to model weights or training data.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from datasets import BEVDataset
from evaluate_exit_lane_v5 import calc_one
from models import TrajectoryModel
import importlib


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def average(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else float('nan')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/v5_splits/val.jsonl')
    p.add_argument('--checkpoint', default='checkpoints/v5_lane_v2_lambda002_best_ade.pt')
    p.add_argument('--alpha-straight', type=float, default=0.5)
    p.add_argument('--alpha-left', type=float, default=1.0)
    p.add_argument('--alpha-right', type=float, default=1.0)
    p.add_argument('--generator-seed', type=int, default=20261021)
    p.add_argument('--max-lines', type=int, default=32)
    p.add_argument('--station-gap-m', type=float, default=1.5)
    p.add_argument('--min-points', type=int, default=5)
    p.add_argument('--center-tolerance-m', type=float, default=.5)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output-dir', default='outputs/v5_nav_scaling')
    args = p.parse_args()
    alphas = {'straight': args.alpha_straight, 'left': args.alpha_left, 'right': args.alpha_right}
    if any(not np.isfinite(a) or a < 0 or a > 1 for a in alphas.values()):
        p.error('All navigation alphas must be finite and within [0, 1]')
    if args.station_gap_m <= 0 or args.min_points < 2:
        p.error('Invalid exit visibility parameters')

    ds = BEVDataset(args.data, split='all', augment=False, max_lines=args.max_lines)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if int(ckpt['horizon']) != ds.path_points:
        raise ValueError('Checkpoint horizon differs from dataset path point count')
    
    # Build the architecture described by the checkpoint.
    variant = ckpt.get("model_variant", "v5")

    if variant == "v5":
        module_name = "models.trajectory_model"
        kwargs = {
            "horizon": int(ckpt["horizon"]),
        }

    elif variant == "v6B1":
        module_name = "models.trajectory_model_v6B1"
        kwargs = {
            "horizon": int(ckpt["horizon"]),
            "bev_encoder": ckpt.get("bev_encoder") or "cnn",
            "bev_token_mode": ckpt.get("bev_token_mode") or "pooled",
        }

    else:
        raise ValueError(f"Unsupported model variant: {variant}")

    module = importlib.import_module(module_name)
    model = module.TrajectoryModel(**kwargs)

    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(args.device)
    model.eval()

    print(
        f"Loaded: variant={variant}, "
        f"encoder={kwargs.get('bev_encoder', 'native')}, "
        f"token_mode={kwargs.get('bev_token_mode', 'native')}, "
        f"epoch={ckpt.get('epoch')}"
    )

    model.eval()

    records = []
    methods = ('reference', 'original', 'nav_scaled')
    with torch.inference_mode():
        for i, row in enumerate(ds.rows):
            item = ds[i]
            inputs = {k: item[k].unsqueeze(0).to(args.device)
                      for k in ('lines', 'line_valid', 'reference', 'navigation')}
            raw = model(inputs)[0].cpu().numpy().astype(np.float64)
            ref = item['reference'].numpy().astype(np.float64)
            gt = item['target_xy'].numpy().astype(np.float64)
            alpha = alphas[row['navigation']]
            predictions = {
                'reference': ref,
                'original': raw,
                'nav_scaled': ref + alpha * (raw - ref),
            }
            # Same GT-station visibility mask regardless of prediction/method.
            for method in methods:
                path = predictions[method]
                error = np.linalg.norm(path - gt, axis=-1)
                strict = calc_one(row, ref, gt, path, args.generator_seed,
                                  args.station_gap_m, args.min_points,
                                  args.center_tolerance_m)
                r = dict(method=method, scene_id=row['id'],
                         base_intersection_id=row['base_intersection_id'],
                         navigation=row['navigation'],
                         ego_distance_m=float(row['metadata']['ego_distance_to_junction_m']),
                         alpha=(0.0 if method == 'reference' else 1.0 if method == 'original' else alpha),
                         path_ade_m=float(error.mean()), path_fde_m=float(error[-1]),
                         exit_eligible=bool(strict['eligible']),
                         visible_target_points=int(strict['visible_target_points']))
                if strict['eligible']:
                    r.update(ref_exit_lat_m=float(strict['reference_lat_mean_m']),
                             exit_lat_m=float(strict['model_lat_mean_m']),
                             center_hit=float(strict['model_center_within_tol_frac']),
                             lane_hit=float(strict['model_within_lane_frac']),
                             heading_deg=float(strict['model_heading_mean_deg']),
                             exit_improvement_vs_ref_m=float(strict['improvement_lat_m']))
                records.append(r)

    summary = []
    for method in methods:
        for nav in ('all', 'straight', 'left', 'right'):
            selected = [r for r in records if r['method'] == method and
                        (nav == 'all' or r['navigation'] == nav)]
            valid = [r for r in selected if r['exit_eligible']]
            s = dict(method=method, navigation=nav, scenes=len(selected),
                     eligible_scenes=len(valid),
                     path_ade_m=average(selected, 'path_ade_m'),
                     path_fde_m=average(selected, 'path_fde_m'),
                     ref_exit_lat_m=average(valid, 'ref_exit_lat_m'),
                     exit_lat_m=average(valid, 'exit_lat_m'),
                     center_hit=average(valid, 'center_hit'),
                     lane_hit=average(valid, 'lane_hit'),
                     heading_deg=average(valid, 'heading_deg'),
                     better_scenes=sum(r['exit_improvement_vs_ref_m'] > 0 for r in valid))
            s['exit_improvement_vs_ref_m'] = s['ref_exit_lat_m'] - s['exit_lat_m']
            summary.append(s)
            print(f'{method:10s} {nav:8s} n={len(selected):3d} eligible={len(valid):3d} '
                  f'ADE={s["path_ade_m"]:.3f}m exit={s["exit_lat_m"]:.3f}m '
                  f'ref_exit={s["ref_exit_lat_m"]:.3f}m '
                  f'center_hit={s["center_hit"]*100:.1f}% '
                  f'better_scenes={s["better_scenes"]}/{len(valid)}', flush=True)
    output = Path(args.output_dir)
    write_csv(output / 'summary.csv', summary)
    write_csv(output / 'per_scene.csv', records)
    print(f'Saved: {output / "summary.csv"} {output / "per_scene.csv"}')
    print('Strict exit metrics are scene-macro means on the same GT-station mask as evaluate_exit_lane_v5.py.')
    print('Keep TEST held out until method selection is complete.')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""V5 exit marking geometry perturbation: fixed token counts, signed response.

Uses ground-truth generator geometry ONLY to select/shift diagnostic inputs and
measure errors; no labels/geometry are injected into the neural network.
Never use perturbation results as a training or inference strategy.
"""
import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch

from datasets import BEVDataset
from tools.model_compat import load_trajectory_model
from evaluate_exit_lane_v5 import (base_geometry, observed_sides, near_visible,
                                   calc_one)

INPUT_KEYS = ('lines', 'line_valid', 'reference', 'navigation')
MODES = ('target', 'nonexit_control')


def elements(row, max_lines):
    return ([(e, False) for e in row['lane_markings']] +
            [(e, True) for e in row['road_boundaries']])[:max_lines]


def target_side(e, anchor, normal, width, ratio):
    if e.get('source_road') != 'exit':
        return False
    pts = np.asarray(e['points'], dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
        return False
    offset = float(np.median((pts - anchor) @ normal))
    return min(abs(offset - width / 2), abs(offset + width / 2)) <= ratio * width


def choose_controls(row_id, candidates, count, seed):
    if count == 0 or len(candidates) < count:
        return []
    raw = hashlib.sha256(f'{seed}|{row_id}'.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(raw[:8], 'big'))
    return sorted(rng.choice(candidates, size=count, replace=False).tolist())


def model_input(item, indices, offset, normal, device):
    batch = {k: item[k].unsqueeze(0).clone().to(device) for k in INPUT_KEYS}
    if offset and indices:
        movement = torch.as_tensor(normal * offset, device=device, dtype=batch['lines'].dtype)
        # Only point positions change. Translation leaves delta features unchanged.
        batch['lines'][0, indices, :, :2] += movement
    return batch


def save_csv(path, records):
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in records for k in r))
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def grouped_mean(records, key):
    return float(np.mean([r[key] for r in records])) if records else float('nan')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/v5_splits/val.jsonl')
    p.add_argument('--checkpoint', default='checkpoints/v5_lane_v2_lambda002_best_ade.pt')
    p.add_argument('--generator-seed', type=int, default=20261021)
    p.add_argument('--max-lines', type=int, default=32)
    p.add_argument('--station-gap-m', type=float, default=1.5)
    p.add_argument('--min-points', type=int, default=5)
    p.add_argument('--center-tolerance-m', type=float, default=0.5)
    p.add_argument('--target-offset-ratio', type=float, default=0.33)
    p.add_argument('--control-seed', type=int, default=42)
    p.add_argument('--deltas', type=float, nargs='+', default=[-0.5, -0.25, 0, 0.25, 0.5])
    p.add_argument('--output-dir', default=None,
                   help='Defaults to outputs/<model>_exit_marking_perturbation')
    p.add_argument('--device', default=('cuda' if torch.cuda.is_available() else 'cpu'))
    args = p.parse_args()
    offsets = sorted(set(round(float(x), 8) for x in args.deltas) | {0.0})
    if args.max_lines < 1 or args.min_points < 2 or args.station_gap_m <= 0:
        p.error('Invalid dimensions or mask configuration')
    if not any(x < 0 for x in offsets) or not any(x > 0 for x in offsets):
        p.error('Need positive and negative --deltas')
    model, ckpt, model_info = load_trajectory_model(args.checkpoint, args.device)
    args.output_dir = args.output_dir or f"outputs/{model_info['model_variant']}_exit_marking_perturbation"
    print(f"Loaded model={model_info['model_variant']} bev_encoder={model_info['bev_encoder']} "
          f"horizon={model_info['horizon']} checkpoint={args.checkpoint}")
    ds = BEVDataset(args.data, split='all', augment=False, max_lines=args.max_lines,
                    ref_points=int(ckpt['horizon']))
    output = []
    per_scene_slopes = []
    available = 0
    with torch.inference_mode():
        for idx, row in enumerate(ds.rows):
            item = ds[idx]
            ref = item['reference'].numpy().astype(float)
            gt = item['target_xy'].numpy().astype(float)
            anchor, tangent, normal, width, _ = base_geometry(row, args.generator_seed)
            all_elems = elements(row, args.max_lines)
            targets = [j for j, (e, _) in enumerate(all_elems)
                       if target_side(e, anchor, normal, width, args.target_offset_ratio)]
            others = [j for j, (e, _) in enumerate(all_elems) if e.get('source_road') != 'exit']
            controls = choose_controls(row['id'], others, len(targets), args.control_seed)
            # Strict original-geometry evaluation mask shared across ALL perturbations.
            sides = observed_sides(row, anchor, tangent, normal, width,
                                   int(row['metadata']['exit_lane_index']))
            stations = (gt - anchor) @ tangent
            mask = ((stations >= 3.) & near_visible(stations, sides[0], args.station_gap_m)
                    & near_visible(stations, sides[1], args.station_gap_m))
            strict = calc_one(row, ref, gt, ref, args.generator_seed,
                              args.station_gap_m, args.min_points, args.center_tolerance_m)
            eligible = bool(strict['eligible'])
            if eligible:
                available += 1
            full = model(model_input(item, [], 0, normal, args.device))[0].cpu().numpy().astype(float)
            for mode, selected in [('target', targets), ('nonexit_control', controls)]:
                signed_by_delta = {}
                for delta in offsets:
                    if delta == 0 or not selected:
                        pred = full
                    else:
                        pred = model(model_input(item, selected, delta, normal, args.device))[0].cpu().numpy().astype(float)
                    signed_diff = (pred - full) @ normal
                    signed_exit = float(signed_diff[mask].mean()) if eligible else float('nan')
                    signed_by_delta[delta] = signed_exit
                    row_out = dict(scene_id=row['id'], navigation=row['navigation'],
                                   distance_m=float(row['metadata']['ego_distance_to_junction_m']),
                                   mode=mode, delta_m=delta, perturbed_lines=len(selected),
                                   target_lines=len(targets), eligible=eligible,
                                   visible_points=int(mask.sum()),
                                   ade_m=float(np.linalg.norm(pred - gt, axis=-1).mean()),
                                   full_ade_m=float(np.linalg.norm(full - gt, axis=-1).mean()))
                    if eligible:
                        center_abs = np.abs((pred[mask] - anchor) @ normal)
                        baseline_abs = np.abs((full[mask] - anchor) @ normal)
                        row_out.update(exit_lat_m=float(center_abs.mean()),
                                       full_exit_lat_m=float(baseline_abs.mean()),
                                       exit_lat_delta_m=float(center_abs.mean()-baseline_abs.mean()),
                                       signed_exit_response_m=signed_exit,
                                       abs_exit_response_m=float(np.abs(signed_diff[mask]).mean()),
                                       exit_xy_shift_m=float(np.linalg.norm((pred-full)[mask],axis=-1).mean()))
                    output.append(row_out)
                if eligible and selected:
                    xx = np.asarray(offsets, dtype=float)
                    yy = np.asarray([signed_by_delta[d] for d in offsets], dtype=float)
                    slope = float(np.dot(xx-xx.mean(), yy-yy.mean()) / np.square(xx-xx.mean()).sum())
                    per_scene_slopes.append(dict(scene_id=row['id'], navigation=row['navigation'],
                                                 mode=mode, perturbed_lines=len(selected),
                                                 slope_m_per_m=slope,
                                                 sign_consistent=bool(slope > 0),
                                                 response_neg05_m=signed_by_delta.get(-0.5, float('nan')),
                                                 response_pos05_m=signed_by_delta.get(0.5, float('nan'))))
    summary = []
    for mode in MODES:
        for nav in ('all', 'straight', 'left', 'right'):
            for delta in offsets:
                selection = [r for r in output if r['mode']==mode and r['delta_m']==delta
                             and r['eligible'] and r['perturbed_lines']>0
                             and (nav=='all' or r['navigation']==nav)]
                if not selection:
                    continue
                summary.append(dict(mode=mode, navigation=nav, delta_m=delta,
                                    scenes=len(selection), mean_perturbed_lines=grouped_mean(selection,'perturbed_lines'),
                                    exit_lat_m=grouped_mean(selection,'exit_lat_m'),
                                    d_exit_m=grouped_mean(selection,'exit_lat_delta_m'),
                                    signed_response_m=grouped_mean(selection,'signed_exit_response_m'),
                                    abs_response_m=grouped_mean(selection,'abs_exit_response_m'),
                                    exit_xy_shift_m=grouped_mean(selection,'exit_xy_shift_m'),
                                    ade_m=grouped_mean(selection,'ade_m')))
    slope_summary = []
    for mode in MODES:
        for nav in ('all', 'straight', 'left', 'right'):
            selection = [r for r in per_scene_slopes if r['mode']==mode and
                         (nav=='all' or r['navigation']==nav)]
            if selection:
                slope_summary.append(dict(mode=mode, navigation=nav, scenes=len(selection),
                                          mean_slope=grouped_mean(selection,'slope_m_per_m'),
                                          median_slope=float(np.median([r['slope_m_per_m'] for r in selection])),
                                          positive_slope_fraction=float(np.mean([r['sign_consistent'] for r in selection]))))
    output_dir = Path(args.output_dir)
    save_csv(output_dir/'per_scene_delta.csv', output)
    save_csv(output_dir/'summary.csv', summary)
    save_csv(output_dir/'slope_per_scene.csv', per_scene_slopes)
    save_csv(output_dir/'slope_summary.csv', slope_summary)
    print(f'Dataset={len(ds)} strictly eligible={available}; offsets={offsets}')
    print('For each condition, n includes ONLY strictly eligible scenes where that type of line was perturbed.')
    for r in slope_summary:
        print(f"{r['mode']:<16} {r['navigation']:<8} n={r['scenes']:3d} "
              f"mean_slope={r['mean_slope']:+.3f} median_slope={r['median_slope']:+.3f} "
              f"positive={100*r['positive_slope_fraction']:.1f}%")
    for r in summary:
        if r['navigation']=='all' and r['delta_m'] != 0:
            print(f"{r['mode']:<16} delta={r['delta_m']:+.2f}m n={r['scenes']:3d} "
                  f"signed_response={r['signed_response_m']:+.3f}m "
                  f"abs_response={r['abs_response_m']:.3f}m d_exit={r['d_exit_m']:+.3f}m")
    print('Saved:', output_dir/'summary.csv', output_dir/'slope_summary.csv')
    print('NOTE: This is an out-of-distribution diagnostic. A positive slope indicates directional sensitivity,')
    print('not proof of causal lane understanding; controls may differ in geometry/location.')


if __name__ == '__main__':
    main()

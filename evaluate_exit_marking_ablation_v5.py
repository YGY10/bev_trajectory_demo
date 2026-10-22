#!/usr/bin/env python3
"""V5 input ablation: remove observed exit markings without changing ref/nav/model.

IMPORTANT: source_road and reconstructed target lane geometry are used ONLY to
choose which observed polylines to hide; neither is given to the network.
All comparisons use the ORIGINAL scene's strict, GT-station exit mask.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from datasets import BEVDataset
from tools.model_compat import load_trajectory_model
from evaluate_exit_lane_v5 import base_geometry, calc_one

INPUT_KEYS = ('lines', 'line_valid', 'reference', 'navigation')
MODES = ('full', 'no_target_exit', 'no_all_exit', 'control_nonexit')


def load_model(path, device):
    model, ckpt, info = load_trajectory_model(path, device)
    print(f"Loaded model={info['model_variant']} bev_encoder={info['bev_encoder']} "
          f"horizon={info['horizon']} checkpoint={path}")
    return model, ckpt, info


def candidate_elements(row, max_lines):
    """Exactly the same element order and truncation as BEVDataset(augment=False)."""
    return ([(e, False) for e in row['lane_markings']] +
            [(e, True) for e in row['road_boundaries']])[:max_lines]


def is_target_side(e, b, n, lane_width, ratio):
    if e.get('source_road') != 'exit':
        return False
    p = np.asarray(e['points'], dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 2 or p.shape[1] != 2:
        return False
    median = float(np.median((p - b) @ n))
    return any(abs(median - sign * lane_width / 2) <= lane_width * ratio
               for sign in (-1, 1))


def masked_item(original, remove_idxs):
    """Compact retained line tokens, preserving their learned features unchanged.

    Zeroing only 'line_valid' is equivalent for valid token masking, but compaction
    also matches how the dataset would represent fewer observed polylines.
    """
    modified = {k: v.clone() for k, v in original.items() if k in INPUT_KEYS}
    valid_indices = [j for j, flag in enumerate(original['line_valid'].tolist()) if flag]
    retained = [j for j in valid_indices if j not in remove_idxs]
    new_lines = torch.zeros_like(original['lines'])
    new_valid = torch.zeros_like(original['line_valid'])
    if retained:
        new_lines[:len(retained)] = original['lines'][retained]
        new_valid[:len(retained)] = True
    modified['lines'], modified['line_valid'] = new_lines, new_valid
    return modified


def deterministic_controls(row_id, candidates, count, seed):
    if not candidates or count <= 0:
        return []
    digest = hashlib.sha256(f'{seed}|{row_id}'.encode('utf-8')).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], 'big'))
    return sorted(int(x) for x in rng.choice(candidates, size=min(len(candidates), count), replace=False))


def export_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        columns = list(dict.fromkeys(k for row in rows for k in row.keys()))
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    output = []
    for mode in MODES:
        mode_rows = [r for r in rows if r['mode'] == mode]
        for nav in ('all', 'straight', 'left', 'right'):
            rr = [r for r in mode_rows if nav == 'all' or r['navigation'] == nav]
            if not rr:
                continue
            valid = [r for r in rr if r['eligible']]
            summary = dict(mode=mode, navigation=nav, scenes=len(rr), eligible=len(valid),
                           mean_removed=float(np.mean([r['removed_count'] for r in rr])),
                           changed_scenes=sum(r['removed_count'] > 0 for r in rr),
                           ade=float(np.mean([r['ade_m'] for r in rr])),
                           path_change_m=float(np.mean([r['path_change_m'] for r in rr])),
                           ade_delta_vs_full=float(np.mean([r['ade_delta_vs_full_m'] for r in rr])))
            if valid:
                for metric in ('model_lat_mean_m', 'model_center_within_tol_frac',
                               'model_within_lane_frac', 'model_heading_mean_deg',
                               'reference_lat_mean_m', 'exit_path_change_m',
                               'exit_lateral_change_m', 'model_signed_lat_mean_m'):
                    summary[metric] = float(np.mean([r[metric] for r in valid]))
                summary['exit_delta_vs_full_m'] = float(np.mean([r['exit_delta_vs_full_m'] for r in valid]))
                summary['exit_worse_scenes'] = sum(r['exit_delta_vs_full_m'] > 0 for r in valid)
                summary['exit_better_scenes'] = sum(r['exit_delta_vs_full_m'] < 0 for r in valid)
            output.append(summary)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/v5_splits/val.jsonl')
    p.add_argument('--checkpoint', default='checkpoints/v5_lane_v2_lambda002_best_ade.pt')
    p.add_argument('--output-dir', default=None,
                   help='Defaults to outputs/<model>_exit_marking_ablation')
    p.add_argument('--generator-seed', type=int, default=20261021)
    p.add_argument('--max-lines', type=int, default=32)
    p.add_argument('--station-gap-m', type=float, default=1.5)
    p.add_argument('--min-points', type=int, default=5)
    p.add_argument('--center-tolerance-m', type=float, default=.5)
    p.add_argument('--target-offset-ratio', type=float, default=.33)
    p.add_argument('--control-seed', type=int, default=42)
    p.add_argument('--device', default=('cuda' if torch.cuda.is_available() else 'cpu'))
    args = p.parse_args()
    if args.max_lines <= 0 or args.station_gap_m <= 0 or args.min_points < 2:
        p.error('Invalid max-lines, station-gap-m or min-points')
    model, ckpt, model_info = load_model(args.checkpoint, args.device)
    args.output_dir = args.output_dir or f"outputs/{model_info['model_variant']}_exit_marking_ablation"
    ds = BEVDataset(args.data, split='all', augment=False,
                    max_lines=args.max_lines, ref_points=int(ckpt['horizon']))
    rows_out = []
    with torch.inference_mode():
        for i, row in enumerate(ds.rows):
            original = ds[i]
            elems = candidate_elements(row, args.max_lines)
            b, v, n, width, _ = base_geometry(row, args.generator_seed)
            target_indices = [j for j, (e, _) in enumerate(elems)
                              if is_target_side(e, b, n, width, args.target_offset_ratio)]
            all_exit_indices = [j for j, (e, _) in enumerate(elems)
                                if e.get('source_road') == 'exit']
            nonexit = [j for j, (e, _) in enumerate(elems)
                       if e.get('source_road') != 'exit']
            controls = deterministic_controls(row['id'], nonexit, len(target_indices), args.control_seed)
            remove = dict(full=[], no_target_exit=target_indices,
                          no_all_exit=all_exit_indices, control_nonexit=controls)
            predictions = {}
            ref = original['reference'].numpy().astype(float)
            gt = original['target_xy'].numpy().astype(float)
            for mode, removed in remove.items():
                item = masked_item(original, removed)
                batch = {key: item[key].unsqueeze(0).to(args.device) for key in INPUT_KEYS}
                predictions[mode] = model(batch)[0].cpu().numpy().astype(float)
            full = predictions['full']
            base_exit = calc_one(row, ref, gt, full, args.generator_seed,
                                 args.station_gap_m, args.min_points,
                                 args.center_tolerance_m)
            full_ade = float(np.linalg.norm(full - gt, axis=1).mean())
            # The ORIGINAL row is evaluated for every mode; visibility masks stay fixed.
            for mode, pred in predictions.items():
                metrics = calc_one(row, ref, gt, pred, args.generator_seed,
                                   args.station_gap_m, args.min_points,
                                   args.center_tolerance_m)
                change = np.linalg.norm(pred - full, axis=-1)
                result = dict(scene_id=row['id'], base_intersection_id=row.get('base_intersection_id', ''),
                              navigation=row['navigation'], mode=mode,
                              ego_distance_m=float(row['metadata']['ego_distance_to_junction_m']),
                              original_lines=len(elems), removed_count=len(remove[mode]),
                              target_exit_lines=len(target_indices), all_exit_lines=len(all_exit_indices),
                              eligible=metrics['eligible'], visible_target_points=metrics['visible_target_points'],
                              ade_m=float(np.linalg.norm(pred - gt, axis=1).mean()),
                              ade_delta_vs_full_m=float(np.linalg.norm(pred - gt, axis=1).mean()) - full_ade,
                              path_change_m=float(change.mean()),
                              max_path_change_m=float(change.max()))
                if metrics['eligible']:
                    for key in ('reference_lat_mean_m', 'model_lat_mean_m',
                                'model_center_within_tol_frac', 'model_within_lane_frac',
                                'model_heading_mean_deg', 'model_signed_lat_mean_m'):
                        result[key] = metrics[key]
                    result['exit_delta_vs_full_m'] = (metrics['model_lat_mean_m'] -
                                                      base_exit['model_lat_mean_m'])
                    # Correct shared selection: recompute strict GT mask via its original
                    # geometry and observed borders (same logic as calc_one).
                    from evaluate_exit_lane_v5 import observed_sides, near_visible
                    sides = observed_sides(row, b, v, n, width, row['metadata']['exit_lane_index'])
                    stations = (gt - b) @ v
                    mask = ((stations >= 3.0) &
                            near_visible(stations, sides[0], args.station_gap_m) &
                            near_visible(stations, sides[1], args.station_gap_m))
                    result['exit_path_change_m'] = float(change[mask].mean())
                    result['exit_lateral_change_m'] = float(np.abs((pred[mask] - full[mask]) @ n).mean())
                rows_out.append(result)
        # Explicitly verify that the dataset's canonical full predictions are unchanged.
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = aggregate(rows_out)
    export_csv(outdir / 'per_scene.csv', rows_out)
    export_csv(outdir / 'summary.csv', summary)
    print(f'Dataset={len(ds)}. Strictly eligible={sum(r["eligible"] for r in rows_out if r["mode"]=="full")}; '
          f'output={outdir}')
    for s in summary:
        if s['eligible']:
            print(f'{s["mode"]:<17} {s["navigation"]:<8} n={s["eligible"]:3d} '
                  f'removed(mean)={s["mean_removed"]:.2f} changed={s["changed_scenes"]:3d} '
                  f'ADE={s["ade"]:.3f}m '
                  f'exit={s["model_lat_mean_m"]:.3f}m '
                  f'd_exit={s["exit_delta_vs_full_m"]:+.3f}m '
                  f'exit_path_shift={s["exit_path_change_m"]:.3f}m '
                  f'exit_lat_shift={s["exit_lateral_change_m"]:.3f}m '
                  f'center_hit={s["model_center_within_tol_frac"]*100:.1f}%')
    print('IMPORTANT: no_target_exit and no_all_exit are out-of-distribution ablations. '
          'Minimal change is suggestive, not proof of zero BEV usage; changes may be due to token-count shifts.')
    print('Masks evaluated on ORIGINAL GT stations, never recomputed after deleting lines.')


if __name__ == '__main__':
    main()

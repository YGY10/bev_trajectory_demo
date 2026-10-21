#!/usr/bin/env python3
"""Strict visible-target-exit-lane evaluation for the V5 multi-ego generator.

Reconstructs target exit centerline from the ORIGINAL generator and seed,
not from GT. Only evaluate GT arc-length points for which BOTH borders of
its target exit lane have observed geometry near the same exit-station.
All lane center/boundary information is used only for evaluation.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from datasets import BEVDataset
from models import TrajectoryModel


def angular_difference(a, b):
    return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def load_rows(path):
    with open(path, encoding='utf8') as f:
        return [json.loads(s) for s in f if s.strip()]


def base_geometry(row, seed):
    # Deliberately reconstruct from the exact generator used to create V5.
    # Exit anchor b, heading v and lane index are not supplied to the model.
    from data.generate_dataset_v5 import make_base
    key = row['base_intersection_id']
    idx = int(key.rsplit('_', 1)[-1])
    base = make_base(idx, seed + 991 * idx)
    meta = row['metadata']
    ok = (base['nav'] == row['navigation'] and
          base['n_out'] == int(meta['exit_lane_count']) and
          base['exit_idx'] == int(meta['exit_lane_index']) and
          abs(base['width_out'] - float(meta['lane_width_exit_m'])) < 1e-4 and
          angular_difference(base['theta'], float(meta['outgoing_heading_rad'])) < 1e-4)
    if not ok:
        raise ValueError(f'{row["id"]}: base geometry does not match generator seed. '
                         'Set --seed to the original V5 generator seed; do not trust metrics otherwise.')
    ego = np.array([-float(meta['ego_distance_to_junction_m']), 0.0])
    return base['b'] - ego, base['v'], base['norm'], base['width_out'], base['exit_idx']


def observed_sides(row, b, v, n, width, target_idx, tolerance_ratio=.33):
    """Collect station samples for both boundaries of *target* exit lane.

    The generator creates boundaries at (j-.5-exit_idx)*lane_width.
    The target lane is between j=exit_idx and j=exit_idx+1.
    """
    stations = {0: [], 1: []}
    for e in row['lane_markings'] + row['road_boundaries']:
        if e.get('source_road') != 'exit':
            continue
        points = np.asarray(e['points'], dtype=float)
        if points.ndim != 2 or len(points) < 2:
            continue
        relative = points - b[None, :]
        station = relative @ v
        lateral = relative @ n
        median = float(np.median(lateral))
        for side, offset in ((0, -width / 2), (1, width / 2)):
            if abs(median - offset) <= width * tolerance_ratio:
                ss = station[np.abs(lateral - offset) < width * tolerance_ratio]
                if len(ss):
                    stations[side].append(ss)
    return {j: np.sort(np.concatenate(a)) if a else np.empty(0) for j, a in stations.items()}


def near_visible(station, observed, max_gap):
    if observed.size == 0:
        return np.zeros(station.shape, dtype=bool)
    ix = np.searchsorted(observed, station)
    left = np.maximum(ix - 1, 0)
    right = np.minimum(ix, len(observed) - 1)
    return np.minimum(np.abs(station - observed[left]),
                      np.abs(station - observed[right])) <= max_gap


def path_errors(pred, target, mask, b, n, v, lane_width, tol_center):
    pred_segment = pred[mask]
    ref_seg = target[mask]
    lateral = np.abs((pred_segment - b) @ n)
    signed = (pred_segment - b) @ n
    # Pointwise fraction within center tolerance and lane physical bounds
    center_hit = float(np.mean(lateral <= tol_center))
    within_lane = float(np.mean(lateral <= lane_width / 2))
    deviation = np.mean(lateral)
    p95 = float(np.percentile(lateral, 95))
    # Local path heading compared against true outgoing roadway heading.
    tangent = np.gradient(pred, axis=0)[mask]
    theta = np.arctan2(tangent[:, 1], tangent[:, 0])
    desired = math.atan2(v[1], v[0])
    heading = np.rad2deg(angular_difference(theta, desired))
    return dict(lat_mean_m=float(deviation), lat_p95_m=p95,
                center_within_tol_frac=center_hit, within_lane_frac=within_lane,
                heading_mean_deg=float(np.mean(heading)),
                signed_lat_mean_m=float(np.mean(signed)))


def calc_one(row, ref, gt, pred, seed, gap, min_points, center_tol):
    b, v, n, width, exit_idx = base_geometry(row, seed)
    sides = observed_sides(row, b, v, n, width, exit_idx)
    station_gt = (gt - b) @ v
    # Physical outgoing lines begin at b+3*v; no invented visibility before b+3m.
    mask = ((station_gt >= 3.0) & near_visible(station_gt, sides[0], gap)
            & near_visible(station_gt, sides[1], gap))
    output = dict(scene_id=row['id'], base_intersection_id=row['base_intersection_id'],
                  navigation=row['navigation'], distance_m=float(row['metadata']['ego_distance_to_junction_m']),
                  observed_exit_lines=int(row['metadata'].get('observed_exit_lines', -1)),
                  visible_left_boundary=bool(len(sides[0])), visible_right_boundary=bool(len(sides[1])),
                  visible_target_points=int(mask.sum()), lane_width_m=float(width))
    if mask.sum() < min_points:
        output.update(eligible=False, reason=f'fewer_than_{min_points}_strict_points')
        return output
    output.update(eligible=True, reason='')
    # Strong invariant: independently reconstructed centerline matches GT on the straight exit.
    gt_lat = np.abs((gt[mask] - b) @ n)
    if float(np.max(gt_lat)) > 0.1:
        raise ValueError(f'{row["id"]}: reconstructed exit centerline differs from GT by '
                         f'{np.max(gt_lat):.3f} m; stop evaluation.')
    output['gt_center_lat_mean_m'] = float(np.mean(gt_lat))
    for prefix, path in [('reference', ref), ('model', pred)]:
        metrics = path_errors(path, gt, mask, b, n, v, width, center_tol)
        output.update({prefix + '_' + key: val for key, val in metrics.items()})
    output['improvement_lat_m'] = (output['reference_lat_mean_m'] - output['model_lat_mean_m'])
    return output


def aggregate(rows, field, value):
    rr = [r for r in rows if r.get('eligible') and (value is None or str(r[field]) == str(value))]
    if not rr:
        return None
    result = dict(group=field, value='all' if value is None else str(value), scenes=len(rr),
                  path_points=sum(x['visible_target_points'] for x in rr))
    keys = ['reference_lat_mean_m', 'model_lat_mean_m', 'reference_heading_mean_deg',
            'model_heading_mean_deg', 'reference_center_within_tol_frac',
            'model_center_within_tol_frac', 'reference_within_lane_frac',
            'model_within_lane_frac', 'improvement_lat_m']
    # Macro average over scenes to prevent long observed intervals from dominating.
    for k in keys:
        result[k] = float(np.mean([r[k] for r in rr]))
    result['model_better_scenes'] = sum(r['improvement_lat_m'] > 0 for r in rr)
    return result


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r.keys()))
    with open(path, 'w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/v5_splits/test.jsonl')
    p.add_argument('--checkpoint', default='checkpoints/best_path_v5.pt')
    p.add_argument('--output-dir', default='outputs/v5_exit_lane_eval')
    p.add_argument('--generator-seed', type=int, default=20261021)
    p.add_argument('--max-lines', type=int, default=32)
    p.add_argument('--station-gap-m', type=float, default=1.5,
                   help='Allow at most this station distance to a measured segment on EACH flank.')
    p.add_argument('--min-points', type=int, default=5,
                   help='Minimum aligned target-exit path samples for eligibility.')
    p.add_argument('--center-tolerance-m', type=float, default=.5)
    p.add_argument('--device', default=None)
    args = p.parse_args()
    if args.station_gap_m <= 0 or args.min_points <= 1: p.error('Invalid threshold')
    import torch
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = TrajectoryModel(horizon=int(ckpt['horizon'])).to(device)
    model.load_state_dict(ckpt['model']);model.eval()
    ds = BEVDataset(args.data, split='all', augment=False, max_lines=args.max_lines,
                    ref_points=int(ckpt['horizon']))
    if len(ds.rows) == 0: raise ValueError('Empty data')
    results = []
    with torch.inference_mode():
        for i, row in enumerate(ds.rows):
            item=ds[i]
            batch={k:v.unsqueeze(0).to(device) for k,v in item.items()}
            prediction=model(batch)[0].cpu().numpy().astype(float)
            ref=item['reference'].numpy().astype(float)
            gt=item['target_xy'].numpy().astype(float)
            results.append(calc_one(row, ref, gt, prediction, args.generator_seed,
                                    args.station_gap_m, args.min_points, args.center_tolerance_m))
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'per_scene.csv', results)
    summary=[]
    for field,values in [('all',[None]),('navigation',['straight','left','right']),
                         ('distance_m',[50.,35.,20.,7.])]:
        for v in values:
            r=aggregate(results,field,v)
            if r:summary.append(r)
    write_csv(out/'summary.csv',summary)
    eligible=sum(r['eligible'] for r in results)
    print(f'Data: {len(results)} | Strictly evaluable: {eligible} | '
          f'Not evaluable: {len(results)-eligible} (no two-sided observed target lane / insufficient path overlap)')
    for r in summary:
        print(f"{r['group']}={r['value']:<8} n={r['scenes']:3d}  "
              f"LAT ref={r['reference_lat_mean_m']:.3f}m model={r['model_lat_mean_m']:.3f}m "
              f"delta={r['improvement_lat_m']:+.3f}m | "
              f"center-hit(0.5m)={r['model_center_within_tol_frac']*100:.1f}% "
              f"lane-hit={r['model_within_lane_frac']*100:.1f}% "
              f"heading={r['model_heading_mean_deg']:.2f}deg")
    print('Saved:', out/'per_scene.csv', out/'summary.csv')
    print('Metrics cover ONLY physically observed target exit straight lane segments. Not all paths/turns.')

if __name__=='__main__': main()

#!/usr/bin/env python3
"""V5 path-error partition: entry, unmarked junction, exit, and strictly visible road segments.

Requires generate_dataset_v5.py and evaluate_exit_lane_v5.py in project root or data/.
Uses generator reconstruction ONLY for evaluation; it is not a model input.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from datasets import BEVDataset
from models import TrajectoryModel
from evaluate_exit_lane_v5 import base_geometry, observed_sides, near_visible
from data.generate_dataset_v5 import make_base


def entry_sides(row, width, tolerance_ratio=0.33):
    observed = {0: [], 1: []}
    for e in row['lane_markings'] + row['road_boundaries']:
        if e.get('source_road') != 'entry':
            continue
        pts = np.asarray(e.get('points', []), dtype=float)
        if pts.ndim != 2 or len(pts) < 2:
            continue
        med = float(np.median(pts[:, 1]))
        for side, offset in ((0, -width / 2), (1, width / 2)):
            if abs(med - offset) <= tolerance_ratio * width:
                keep = np.abs(pts[:, 1] - offset) < tolerance_ratio * width
                if keep.any():
                    observed[side].append(pts[keep, 0])
    return {k: np.sort(np.concatenate(v)) if v else np.empty(0) for k, v in observed.items()}


def errors(pred, ref, gt, mask):
    delta_pred = pred[mask] - gt[mask]
    delta_ref = ref[mask] - gt[mask]
    ep = np.linalg.norm(delta_pred, axis=1)
    er = np.linalg.norm(delta_ref, axis=1)
    tangent = np.gradient(gt, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.stack((-tangent[:, 1], tangent[:, 0]), axis=1)
    nerr = np.abs(np.sum((pred - gt) * normal, axis=1))[mask]
    nref = np.abs(np.sum((ref - gt) * normal, axis=1))[mask]
    residual = np.linalg.norm(pred - ref, axis=1)[mask]
    return dict(points=int(mask.sum()), model_euclidean_m=float(ep.mean()),
                reference_euclidean_m=float(er.mean()), improvement_euclidean_m=float(er.mean()-ep.mean()),
                model_normal_m=float(nerr.mean()), reference_normal_m=float(nref.mean()),
                model_residual_m=float(residual.mean()))


def write_csv(path, rows):
    if not rows: return
    with open(path, 'w', newline='', encoding='utf8') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/v5_splits/test.jsonl')
    p.add_argument('--checkpoint', default='checkpoints/best_path_v5.pt')
    p.add_argument('--output-dir', default='outputs/v5_region_analysis')
    p.add_argument('--generator-seed', type=int, default=20261021)
    p.add_argument('--max-lines', type=int, default=32)
    p.add_argument('--station-gap-m', type=float, default=1.5)
    p.add_argument('--min-points', type=int, default=5)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args=p.parse_args()

    checkpoint=torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model=TrajectoryModel(horizon=int(checkpoint['horizon'])).to(args.device)
    model.load_state_dict(checkpoint['model']);model.eval()
    ds=BEVDataset(args.data,split='all',augment=False,max_lines=args.max_lines,
                  ref_points=int(checkpoint['horizon']))
    all_rows=[]
    with torch.inference_mode():
        for i, row in enumerate(ds.rows):
            sample=ds[i]
            batch={k:v.unsqueeze(0).to(args.device) for k,v in sample.items()}
            pred=model(batch)[0].cpu().numpy().astype(float)
            ref=sample['reference'].numpy().astype(float)
            gt=sample['target_xy'].numpy().astype(float)
            b,v,n,width,exit_idx=base_geometry(row,args.generator_seed)
            idx=int(row['base_intersection_id'].rsplit('_',1)[-1])
            base=make_base(idx,args.generator_seed+991*idx)
            distance=float(row['metadata']['ego_distance_to_junction_m'])
            # In ego coordinates, the centerline of incoming road lies on y=0.
            # Entry markings physically end at global x=-2.
            entry_physical=gt[:,0] <= (distance-2.0)
            exit_station=(gt-b)@v
            exit_lateral=np.abs((gt-b)@n)
            # The outbound straight begins at exit anchor + 3m along v.
            exit_physical=(exit_station>=3.0)&(exit_lateral<0.10)
            # Physical interior: not covered by the two declared road segments.
            junction_physical=~(entry_physical|exit_physical)
            # Reconstruct true *observed* two-flank coverage using sensor polyline samples.
            el=entry_sides(row,float(base['width_in']))
            entry_visible=(entry_physical & near_visible(gt[:,0],el[0],args.station_gap_m)
                           & near_visible(gt[:,0],el[1],args.station_gap_m))
            xs=observed_sides(row,b,v,n,width,exit_idx)
            exit_visible=(exit_physical & near_visible(exit_station,xs[0],args.station_gap_m)
                          & near_visible(exit_station,xs[1],args.station_gap_m))
            visible=entry_visible|exit_visible
            masks={
                'entry_physical':entry_physical,
                'junction_unmarked_physical':junction_physical,
                'exit_physical':exit_physical,
                'entry_two_sides_visible':entry_visible,
                'exit_two_sides_visible':exit_visible,
                'visible_two_sides_either':visible,
                'not_two_sides_visible':~visible,
            }
            for zone,mask in masks.items():
                if mask.sum()<args.min_points:continue
                record=dict(scene_id=row['id'],navigation=row['navigation'],distance_m=distance,
                            zone=zone,**errors(pred,ref,gt,mask))
                all_rows.append(record)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'per_scene_region.csv',all_rows)
    summary=[]
    for group in ['all','navigation','distance_m']:
        groups=sorted(set(str(r[group]) for r in all_rows)) if group!='all' else ['all']
        for gv in groups:
            subset=[r for r in all_rows if group=='all' or str(r[group])==gv]
            for zone in ['entry_physical','junction_unmarked_physical','exit_physical',
                         'entry_two_sides_visible','exit_two_sides_visible',
                         'visible_two_sides_either','not_two_sides_visible']:
                rr=[r for r in subset if r['zone']==zone]
                if not rr:continue
                d={'group':group,'value':gv,'zone':zone,'scenes':len(rr),'points':sum(r['points'] for r in rr)}
                for key in ['model_euclidean_m','reference_euclidean_m','improvement_euclidean_m',
                            'model_normal_m','reference_normal_m','model_residual_m']:
                    d[key]=float(np.mean([r[key] for r in rr]))
                d['model_worse_scenes']=sum(r['improvement_euclidean_m']<0 for r in rr)
                summary.append(d)
    write_csv(out/'summary.csv',summary)
    print('Test scenes:',len(ds),'| Region records:',len(all_rows))
    print('NOTE: region metrics are macro-averaged over eligible scenes (>=5 points); '
          'different regions may contain different sets of scenes.')
    for r in summary:
        if r['group']=='all':
            print(f"{r['zone']:<28} n={r['scenes']:3d} "
                  f"Ref={r['reference_euclidean_m']:.3f}m "
                  f"Model={r['model_euclidean_m']:.3f}m "
                  f"Normal={r['model_normal_m']:.3f}m "
                  f"Improvement={r['improvement_euclidean_m']:+.3f}m")
    print('Saved:',out/'per_scene_region.csv',out/'summary.csv')


if __name__=='__main__':main()

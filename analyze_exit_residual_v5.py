#!/usr/bin/env python3
"""V5 exit-lane signed residual diagnostics (no training or checkpoint changes).

Uses the same GT-indexed, physically observed two-sided target-exit mask as
`evaluate_exit_lane_v5.py`. Reports signed correction direction, magnitude,
undershoot/overshoot, and scene-balanced outcomes for multiple checkpoints.
Labels such as `wrong_direction` refer to points with non-negligible reference
lateral error, not target-lane detection or semantic association.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from datasets import BEVDataset
from models import TrajectoryModel
from evaluate_exit_lane_v5 import (load_rows, base_geometry, observed_sides,
                                   near_visible)


def write_rows(path, rows):
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def load_model(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    model = TrajectoryModel(horizon=int(ck['horizon'])).to(device)
    model.load_state_dict(ck['model'], strict=True)
    model.eval()
    return model


def group_summary(rows, group, value, threshold):
    rr = [r for r in rows if group == 'all' or str(r[group]) == str(value)]
    if not rr: return None
    # Point-wise categories are built only over reference errors beyond threshold.
    strong = [r for r in rr if r['reference_significant']]
    eps = 1e-9
    return dict(model=rr[0]['model'], group=group, value=str(value), scenes=len(set(r['scene_id'] for r in rr)),
                points=len(rr), significant_reference_points=len(strong),
                ref_abs_lat_m=float(np.mean([r['ref_abs_lat_m'] for r in rr])),
                pred_abs_lat_m=float(np.mean([r['pred_abs_lat_m'] for r in rr])),
                mean_improvement_m=float(np.mean([r['improvement_m'] for r in rr])),
                mean_signed_correction_m=float(np.mean([r['correction_signed_m'] for r in rr])),
                mean_abs_correction_m=float(np.mean([abs(r['correction_signed_m']) for r in rr])),
                beneficial_direction_rate=float(np.mean([r['beneficial_direction'] for r in strong])) if strong else float('nan'),
                wrong_direction_rate=float(np.mean([r['wrong_direction'] for r in strong])) if strong else float('nan'),
                overshoot_rate=float(np.mean([r['overshoot'] for r in strong])) if strong else float('nan'),
                overcorrection_rate=float(np.mean([r['overcorrection'] for r in strong])) if strong else float('nan'),
                insufficient_correction_rate=float(np.mean([r['insufficient'] for r in strong])) if strong else float('nan'))


def scene_summary(points):
    sample = points[0]
    out = {k:sample[k] for k in ('model','scene_id','base_intersection_id','navigation','distance_m')}
    out['valid_points'] = len(points)
    out['reference_abs_lat_m'] = float(np.mean([p['ref_abs_lat_m'] for p in points]))
    out['prediction_abs_lat_m'] = float(np.mean([p['pred_abs_lat_m'] for p in points]))
    out['improvement_m'] = out['reference_abs_lat_m'] - out['prediction_abs_lat_m']
    strong = [p for p in points if p['reference_significant']]
    out['significant_ref_points'] = len(strong)
    for name in ('beneficial_direction','wrong_direction','overshoot','overcorrection','insufficient'):
        out[name+'_frac'] = float(np.mean([p[name] for p in strong])) if strong else float('nan')
    out['max_abs_prediction_lat_m'] = float(max(p['pred_abs_lat_m'] for p in points))
    return out


def print_macro(name, rr, group_name='all', group_val='all'):
    if not rr:return
    ref = np.mean([r['reference_abs_lat_m'] for r in rr]); pred = np.mean([r['prediction_abs_lat_m'] for r in rr])
    useful = [r for r in rr if r['significant_ref_points']]
    beneficial = np.mean([r['beneficial_direction_frac'] for r in useful]) if useful else float('nan')
    wrong = np.mean([r['wrong_direction_frac'] for r in useful]) if useful else float('nan')
    overs = np.mean([r['overshoot_frac'] for r in useful]) if useful else float('nan')
    better = sum(r['improvement_m']>0 for r in rr)
    print(f'{name:<26} {group_name}={group_val:<8} n={len(rr):3d} '
          f'ref={ref:.3f}m pred={pred:.3f}m delta={ref-pred:+.3f}m '
          f'better_scenes={better}/{len(rr)} beneficial={beneficial:.1%} '
          f'wrong={wrong:.1%} overshoot={overs:.1%}', flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',default='data/v5_splits/val.jsonl')
    ap.add_argument('--checkpoint',action='append',nargs=2,metavar=('LABEL','FILE'),required=True,
                    help='Repeat --checkpoint label path for each model')
    ap.add_argument('--max-lines',type=int,default=32)
    ap.add_argument('--generator-seed',type=int,default=20261021)
    ap.add_argument('--station-gap-m',type=float,default=1.5)
    ap.add_argument('--min-points',type=int,default=5)
    ap.add_argument('--deadband-m',type=float,default=0.05,
                    help='Ignore near-centered references when classifying correction direction')
    ap.add_argument('--correction-eps-m',type=float,default=0.01,
                    help='Treat lateral corrections below this as no significant movement')
    ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--output-dir',default='outputs/v5_exit_residual_analysis')
    args=ap.parse_args()
    if args.deadband_m<0 or args.correction_eps_m<0 or args.min_points<1 or args.station_gap_m<=0:
        ap.error('Invalid threshold')
    labels=[x[0] for x in args.checkpoint]
    if len(set(labels))!=len(labels):ap.error('Each checkpoint label must be unique')
    ds=BEVDataset(args.data,split='all',augment=False,max_lines=args.max_lines)
    models={label:load_model(path,args.device) for label,path in args.checkpoint}
    for name,model in models.items():
        if model.horizon != ds.path_points: raise ValueError(f'{name}: checkpoint horizon mismatch')
    rows=[];excluded=0; scene_rows=[]
    for index,row in enumerate(ds.rows):
        item=ds[index]
        ref=item['reference'].numpy().astype(np.float64)
        gt=item['target_xy'].numpy().astype(np.float64)
        b,v,n,width,_=base_geometry(row,args.generator_seed)
        sides=observed_sides(row,b,v,n,width,row['metadata']['exit_lane_index'])
        s=(gt-b)@v
        mask=(s>=3.0)&near_visible(s,sides[0],args.station_gap_m)&near_visible(s,sides[1],args.station_gap_m)
        ids=np.flatnonzero(mask)
        if len(ids)<args.min_points:
            excluded+=1;continue
        if np.max(np.abs((gt[mask]-b)@n))>0.1:
            raise ValueError(f'{row["id"]}: GT is not centered on reconstructed exit straight')
        batch={k:item[k].unsqueeze(0).to(args.device) for k in ('lines','line_valid','reference','navigation')}
        preds={}
        with torch.inference_mode():
            for label,model in models.items():
                preds[label]=model(batch)[0].cpu().numpy().astype(np.float64)
        ref_signed=(ref-b)@n
        for label,pred in preds.items():
            pred_signed=(pred-b)@n
            current=[]
            for idx in ids:
                a=float(ref_signed[idx]); p=float(pred_signed[idx]); c=p-a; needed=-a
                significant=abs(a)>args.deadband_m
                # Corrections taking the opposite direction from the needed move are 'wrong'.
                beneficial=significant and (c*needed>0) and (abs(c)>args.correction_eps_m)
                wrong=significant and (c*needed<0) and (abs(c)>args.correction_eps_m)
                # Overshoot means crossing the centerline; overcorrection is a stricter
                # subset where doing so makes absolute lateral error worse than reference.
                overshoot=significant and (a*p<0)
                overcorrection=overshoot and (abs(p)>abs(a))
                insufficient=significant and beneficial and (not overshoot)
                point=dict(model=label,scene_id=row['id'],base_intersection_id=row['base_intersection_id'],
                    navigation=row['navigation'],distance_m=float(row['metadata']['ego_distance_to_junction_m']),
                    point_index=int(idx),station_m=float(s[idx]),lane_width_m=float(width),
                    ref_signed_lat_m=a,pred_signed_lat_m=p,
                    desired_correction_m=needed,correction_signed_m=c,
                    ref_abs_lat_m=abs(a),pred_abs_lat_m=abs(p),improvement_m=abs(a)-abs(p),
                    reference_significant=bool(significant),beneficial_direction=bool(beneficial),
                    wrong_direction=bool(wrong),overshoot=bool(overshoot),
                    overcorrection=bool(overcorrection),insufficient=bool(insufficient))
                rows.append(point);current.append(point)
            scene_rows.append(scene_summary(current))
    out=Path(args.output_dir)
    write_rows(out/'per_point.csv',rows)
    write_rows(out/'per_scene.csv',scene_rows)
    summaries=[]
    for label in labels:
        pts=[p for p in rows if p['model']==label]
        sc=[p for p in scene_rows if p['model']==label]
        print(f'\n{label}: eligible={len(sc)}/{len(ds)} point_count={len(pts)} excluded={excluded}')
        print_macro(label,sc)
        for nav in ('straight','left','right'):
            print_macro(label,[r for r in sc if r['navigation']==nav],'navigation',nav)
        for group,value in [('all','all')]+[('navigation',x) for x in ('straight','left','right')]+[('distance_m',x) for x in (20.,7.)]:
            summary=group_summary(pts,group,value,args.deadband_m)
            if summary: summaries.append(summary)
    write_rows(out/'point_weighted_summary.csv',summaries)
    print('\nSaved:',out/'per_point.csv',out/'per_scene.csv',out/'point_weighted_summary.csv')
    print('Scene macro statistics above weight each intersection-pose equally; direction rates are '
          'computed on points with |reference lateral error| above deadband.')
    print('IMPORTANT: mask uses GT station like strict evaluator. Results measure signed '
          'lateral corrections on visible exit straights, not longitudinal error or lane-choice ability.')

if __name__=='__main__': main()

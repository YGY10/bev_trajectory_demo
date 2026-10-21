#!/usr/bin/env python3
"""Preflight audit for synthetic V5 intersections. No torch/GPU needed.

This is a geometry/serialization check, NOT a driving-safety or topology validator.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def points(x):
    p = np.asarray(x, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all():
        return None
    return p


def min_dist(a, b):
    # Distance from each point a to dense points b; suitable for quick audit, not exact segment distances.
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1).min(axis=1)


def evaluate(row, capacity):
    markings = row.get('lane_markings', [])
    boundaries = row.get('road_boundaries', [])
    elems = markings + boundaries
    gt = points(row.get('gt_path', []))
    ref = points(row.get('reference_line', []))
    meta = row.get('metadata', {})
    width = float(meta.get('lane_width_m', 3.5))
    out = dict(id=row.get('id', 'unknown'), navigation=row.get('navigation', 'unknown'),
               entry=meta.get('entry_lane', '?'), exit=meta.get('exit_lane', '?'),
               markings=len(markings), boundaries=len(boundaries), total_lines=len(elems),
               excess_lines=max(0, len(elems)-capacity),
               drop_fraction=round(max(0,len(elems)-capacity)/max(len(elems),1), 3),
               bad_polylines=0, zero_length_polylines=0, gt_points=0,
               gt_length_m=None, gt_spacing_max_err_m=None, ref_max_dist_to_gt_m=None,
               exit_corridor_lines=0, exit_corridor_markings=0,
               visible_outgoing_length_m=0.0, warnings='')
    warn=[]
    valid=[]
    for e in elems:
        p=points(e.get('points', []))
        if p is None:
            out['bad_polylines']+=1;continue
        seg=np.linalg.norm(np.diff(p,axis=0),axis=1)
        if seg.sum()<0.1:out['zero_length_polylines']+=1
        valid.append((p, e in markings))
    if gt is None or ref is None:
        warn.append('invalid_GTor_reference')
    else:
        seg=np.linalg.norm(np.diff(gt,axis=0),axis=1)
        out['gt_points']=len(gt)
        out['gt_length_m']=round(float(seg.sum()),3)
        out['gt_spacing_max_err_m']=round(float(abs(seg-1).max()),3)
        # Use closest GT sample for a coarse geometric distortion check.
        out['ref_max_dist_to_gt_m']=round(float(min_dist(ref,gt).max()),3)
        if len(gt)!=81 or out['gt_spacing_max_err_m']>0.15:
            warn.append('gt_81_points_or_spacing_issue')
        if np.linalg.norm(gt[0])>0.1:warn.append('gt_start_not_ego_origin')
        if len(elems)>capacity:warn.append('exceeds_capacity')
        # Looking for segments alongside the outgoing path, not any lines near the entrance.
        # gt progress >=40m is only a heuristic; 9-sample generator does not label polyline ownership.
        forward_heading=float(meta.get('outgoing_heading_rad',0.))
        forward=np.array([math.cos(forward_heading), math.sin(forward_heading)])
        g=gt[min(40,len(gt)-1):]
        for p,is_marking in valid:
            if len(p)<2:continue
            tangent=p[-1]-p[0]
            length=np.linalg.norm(tangent)
            if length<1e-5:continue
            aligned=abs(float(np.dot(tangent/length,forward))) >= math.cos(math.radians(25))
            proximity=(min_dist(p,g) < 1.25*width).sum()
            if aligned and proximity>=3:
                out['exit_corridor_lines']+=1
                out['exit_corridor_markings']+=int(is_marking)
                out['visible_outgoing_length_m']=round(max(out['visible_outgoing_length_m'],float(length)),2)
        if out['exit_corridor_markings']<1:warn.append('no_exit_marking_detected_heuristic')
        if out['exit_corridor_lines']<2:warn.append('little_exit_geometry_detected_heuristic')
    if out['bad_polylines']:warn.append('invalid_polyline')
    if out['zero_length_polylines']:warn.append('degenerate_polyline')
    out['warnings']=';'.join(warn)
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data', default='data/intersection_v5_fixed.jsonl')
    ap.add_argument('--max-lines', type=int, default=16)
    ap.add_argument('--output', default='outputs/v5_audit/report.csv')
    args=ap.parse_args()
    with Path(args.data).open(encoding='utf-8') as f:
        rows=[json.loads(line) for line in f if line.strip()]
    if not rows:raise SystemExit('Empty data file')
    results=[evaluate(row,args.max_lines) for row in rows]
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=results[0].keys());writer.writeheader();writer.writerows(results)
    print(f'Scenes: {len(rows)}; max_lines={args.max_lines}')
    print(f'Lines: min={min(r["total_lines"] for r in results)}, mean={np.mean([r["total_lines"] for r in results]):.2f}, max={max(r["total_lines"] for r in results)}')
    print(f'Over limit: {sum(r["excess_lines"]>0 for r in results)}/{len(rows)}')
    for nav in ('straight','left','right'):
        group=[r for r in results if r['navigation']==nav]
        if group:
            print(f'{nav:<8}: n={len(group)} mean_lines={np.mean([r["total_lines"] for r in group]):.1f} '
                  f'no_exit_marking_heuristic={sum("no_exit_marking_detected_heuristic" in r["warnings"] for r in group)}')
    print('Flagged scenes:')
    for r in results:
        if r['warnings']:print(' ',r['id'],r['warnings'])
    print('CSV:',dest)
    print('Note: exit visibility is a coarse geometric heuristic, not a validated lane-to-lane topology check.')


if __name__=='__main__':main()

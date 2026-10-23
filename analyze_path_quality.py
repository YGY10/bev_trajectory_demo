#!/usr/bin/env python3
"""V5/V6B1G checkpoint path quality audit; evaluation only."""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from datasets.bev_dataset import BEVDataset
from tools.model_compat import load_trajectory_model
from evaluate_exit_lane_v5 import calc_one


def write_csv(path, rows):
    if not rows: return
    cols=list(dict.fromkeys(k for r in rows for k in r))
    with open(path,'w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f, fieldnames=cols);w.writeheader();w.writerows(rows)


def mean(rows,k):
    a=[r[k] for r in rows if k in r and np.isfinite(r[k])]
    return float(np.mean(a)) if a else float('nan')


def plot_scene(path,row,ref,gt,pred,record):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,7))
    for e in row['road_boundaries']:
        pts=np.asarray(e['points']);ax.plot(pts[:,0],pts[:,1],color='#515969' if e.get('source_road')=='exit' else '#a9aeb8',lw=1.2,alpha=.8)
    for e in row['lane_markings']:
        pts=np.asarray(e['points']);ax.plot(pts[:,0],pts[:,1],color='#c5a14d' if e.get('color')=='yellow' else '#71a8c7',lw=1.2,alpha=.8)
    ax.plot(ref[:,0],ref[:,1],'--',c='#d9871a',lw=2,label='Reference')
    ax.plot(gt[:,0],gt[:,1],'-',c='#188c68',lw=2.5,label='GT')
    ax.plot(pred[:,0],pred[:,1],'-',c='#cb3347',lw=2,label='Prediction')
    i=int(np.linalg.norm(pred-gt,axis=-1).argmax())
    ax.plot([gt[i,0],pred[i,0]],[gt[i,1],pred[i,1]],':',color='#cc2336',lw=1.5)
    ax.scatter([0],[0],marker='^',s=65,c='black',zorder=6,label='Ego')
    ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
    ax.set_xlabel('x: forward (m)');ax.set_ylabel('y: left (m)')
    ax.set_title(f"{row['id']} | ADE {record['model_ade_m']:.3f}m | Ref {record['reference_ade_m']:.3f}m | FDE {record['model_fde_m']:.3f}m",fontsize=9)
    ax.legend(loc='best',fontsize=8);fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--data',default='data/v5_splits/val.jsonl')
    p.add_argument('--output-dir',default='outputs/v6B1G_path_quality')
    p.add_argument('--max-lines',type=int,default=32)
    p.add_argument('--device',default=None)
    p.add_argument('--generator-seed',type=int,default=20261021)
    p.add_argument('--station-gap-m',type=float,default=1.5)
    p.add_argument('--min-points',type=int,default=5)
    p.add_argument('--center-tol-m',type=float,default=.5)
    p.add_argument('--top-k',type=int,default=10)
    p.add_argument('--plot-count',type=int,default=20)
    a=p.parse_args()
    device=a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model,ck,info=load_trajectory_model(a.checkpoint,device)
    ds=BEVDataset(a.data,split='all',augment=False,max_lines=a.max_lines,ref_points=int(info['horizon']))
    if ds.path_points!=int(info['horizon']):raise ValueError('Horizon mismatch')
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    records=[];scene_data={}
    for idx,row in enumerate(ds.rows):
        item=ds[idx]
        batch={k:v.unsqueeze(0).to(device) for k,v in item.items()}
        with torch.inference_mode():pred=model(batch)[0].detach().cpu().numpy().astype(float)
        ref=item['reference'].numpy().astype(float)
        gt=item['target_xy'].numpy().astype(float)
        if not np.isfinite(pred).all():raise RuntimeError(f'Nonfinite prediction: {row["id"]}')
        err=np.linalg.norm(pred-gt,axis=-1);referr=np.linalg.norm(ref-gt,axis=-1)
        n=len(err);b1=n//3;b2=2*n//3
        correction=np.linalg.norm(pred-ref,axis=-1)
        spacing=np.linalg.norm(np.diff(pred,axis=0),axis=-1)
        metadata=row.get('metadata',{})
        strict=calc_one(row,ref,gt,pred,a.generator_seed,a.station_gap_m,a.min_points,a.center_tol_m)
        r=dict(
            scene_id=row['id'],base_intersection_id=row.get('base_intersection_id',''),
            navigation=row['navigation'],distance_m=float(metadata['ego_distance_to_junction_m']),
            exit_visible=bool(strict['observed_exit_lines']>0),
            observed_exit_lines=strict['observed_exit_lines'],
            strict_exit_eligible=bool(strict['eligible']),
            strict_visible_target_points=int(strict['visible_target_points']),
            model_ade_m=float(err.mean()),reference_ade_m=float(referr.mean()),
            improvement_ade_m=float(referr.mean()-err.mean()),
            model_fde_m=float(err[-1]),reference_fde_m=float(referr[-1]),
            improvement_fde_m=float(referr[-1]-err[-1]),
            model_early_ade_m=float(err[:b1].mean()),
            model_middle_ade_m=float(err[b1:b2].mean()),model_late_ade_m=float(err[b2:].mean()),
            model_max_point_error_m=float(err.max()),worst_station_index=int(err.argmax()),
            model_spacing_error_m=float(np.mean(abs(spacing-ds.spacing))),
            correction_mean_m=float(correction.mean()),correction_end_m=float(correction[-1]),
        )
        if strict['eligible']:
            r.update(strict_model_lat_mean_m=strict['model_lat_mean_m'],
                     strict_reference_lat_mean_m=strict['reference_lat_mean_m'],
                     strict_improvement_lat_m=strict['improvement_lat_m'],
                     strict_model_center_hit=strict['model_center_within_tol_frac'],
                     strict_model_lane_hit=strict['model_within_lane_frac'],
                     strict_model_heading_mean_deg=strict['model_heading_mean_deg'])
        records.append(r);scene_data[row['id']]=(row,ref,gt,pred,r)
        if (idx+1)%50==0:print(f'Processed {idx+1}/{len(ds)}',flush=True)
    write_csv(out/'per_scene.csv',records)
    groups=defaultdict(list)
    for r in records:
        groups[('ALL','all')].append(r)
        groups[('NAV',r['navigation'])].append(r)
        groups[('DISTANCE',f'{r["distance_m"]:g}')].append(r)
        groups[('EXIT_VISIBLE',str(r['exit_visible']))].append(r)
        groups[('NAV_DISTANCE',f'{r["navigation"]}_d{r["distance_m"]:g}')].append(r)
    basic=['model_ade_m','reference_ade_m','improvement_ade_m','model_fde_m',
           'reference_fde_m','model_early_ade_m','model_middle_ade_m','model_late_ade_m',
           'model_spacing_error_m','correction_mean_m','correction_end_m']
    strict_cols=['strict_model_lat_mean_m','strict_reference_lat_mean_m',
                 'strict_improvement_lat_m','strict_model_center_hit',
                 'strict_model_lane_hit','strict_model_heading_mean_deg']
    summary=[]
    for (group,value),rows in sorted(groups.items()):
        eligible=[r for r in rows if r['strict_exit_eligible']]
        result={'group':group,'value':value,'count':len(rows),
                'model_better_fraction':float(np.mean([r['improvement_ade_m']>0 for r in rows])),
                'strict_exit_count':len(eligible)}
        result.update({k:mean(rows,k) for k in basic})
        result.update({k:mean(eligible,k) for k in strict_cols})
        summary.append(result)
    write_csv(out/'summary.csv',summary)
    ranks={
        'worst_ade':('model_ade_m',True,lambda r:True),
        'worst_fde':('model_fde_m',True,lambda r:True),
        'worse_than_reference':('improvement_ade_m',False,lambda r:True),
        'worst_late':('model_late_ade_m',True,lambda r:True),
        'worst_strict_exit':('strict_model_lat_mean_m',True,lambda r:r['strict_exit_eligible']),
    }
    ranked=[];plot_ids=[]
    for category,(metric,reverse,filter_fn) in ranks.items():
        chosen=sorted((r for r in records if filter_fn(r)),key=lambda r:r[metric],reverse=reverse)
        for rank,r in enumerate(chosen[:a.top_k],1):
            ranked.append({'category':category,'rank':rank,**r})
            if r['scene_id'] not in plot_ids:plot_ids.append(r['scene_id'])
    write_csv(out/'failure_ranking.csv',ranked)
    plot_dir=out/'plots';plot_dir.mkdir(exist_ok=True)
    plots=[]
    for scene_id in plot_ids[:a.plot_count]:
        row,ref,gt,pred,r=scene_data[scene_id]
        image=plot_dir/(scene_id+'.png');plot_scene(image,row,ref,gt,pred,r)
        plots.append({'scene_id':scene_id,'image':str(image)})
    write_csv(out/'plots_index.csv',plots)
    (out/'run_info.json').write_text(json.dumps({'checkpoint':a.checkpoint,'model_variant':info['model_variant'],
        'epoch':ck.get('epoch'),'data':a.data,'scenes':len(records),'generator_seed':a.generator_seed,
        'station_gap_m':a.station_gap_m,'min_points':a.min_points,'center_tolerance_m':a.center_tol_m,
        'note':'Synthetic GT; no human driving claims'},indent=2),encoding='utf8')
    print('\n=== Path quality ===')
    for r in summary:
        if r['group'] in ('ALL','NAV','DISTANCE','EXIT_VISIBLE'):
            print(f"{r['group']:12s} {r['value']:>9s} n={r['count']:3d} "
                  f"ADE={r['model_ade_m']:.3f} ref={r['reference_ade_m']:.3f} "
                  f"gain={r['improvement_ade_m']:+.3f} FDE={r['model_fde_m']:.3f} "
                  f"strict_n={r['strict_exit_count']} strict_lat={r['strict_model_lat_mean_m']:.3f}")
    print(f'CSV: {out}/per_scene.csv, summary.csv, failure_ranking.csv')
    print(f'Plots: {len(plots)} in {plot_dir}')


if __name__=='__main__':main()

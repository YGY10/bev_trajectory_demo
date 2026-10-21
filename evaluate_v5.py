"""Reference baseline and optional model comparison, stratified by ego distance and exit visibility."""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from datasets import BEVDataset


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True)
    ap.add_argument('--checkpoint',default=None,help='Optional V4/V5 checkpoint to evaluate')
    ap.add_argument('--output-dir',default='outputs/v5_eval')
    ap.add_argument('--device',default=None)
    args=ap.parse_args()
    ds=BEVDataset(args.data,split='all',augment=False, max_lines=32)
    model=None;torch=None;device=None
    if args.checkpoint:
        import torch as T
        from models import TrajectoryModel
        torch=T;device=args.device or ('cuda' if T.cuda.is_available() else 'cpu')
        ckpt=T.load(args.checkpoint,map_location='cpu',weights_only=True)
        model=TrajectoryModel(horizon=ds.path_points).to(device)
        model.load_state_dict(ckpt['model']);model.eval()
    records=[]
    for i,row in enumerate(ds.rows):
        item=ds[i]
        target=item['target_xy'].numpy()
        ref=item['reference'].numpy()
        m=row.get('metadata',{})
        dist=float(m.get('ego_distance_to_junction_m',float('nan')))
        visible=int(m.get('observed_exit_lines',-1))
        entry={'scene_id':row['id'],'base_intersection_id':row.get('base_intersection_id',m.get('base_intersection_id','')),
               'navigation':row['navigation'],'ego_distance_m':dist,'observed_exit_lines':visible}
        for name,pred in [('reference',ref)]:
            e=np.linalg.norm(pred-target,axis=-1)
            entry[f'{name}_mean_m']=float(e.mean());entry[f'{name}_end_m']=float(e[-1])
        if model:
            batch={k:v.unsqueeze(0).to(device) for k,v in item.items()}
            with torch.inference_mode():pred=model(batch)[0].cpu().numpy()
            e=np.linalg.norm(pred-target,axis=-1)
            entry['model_mean_m']=float(e.mean());entry['model_end_m']=float(e[-1])
            entry['improvement_m']=entry['reference_mean_m']-entry['model_mean_m']
        records.append(entry)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    with (out/'per_scene.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    group=defaultdict(list)
    for r in records:
        group[('ALL','all')].append(r)
        group[('DISTANCE',str(round(r['ego_distance_m'],1)))].append(r)
        group[('NAV',r['navigation'])].append(r)
        group[('VISIBILITY','none' if r['observed_exit_lines']==0 else 'visible')].append(r)
    summary=[]
    cols=['reference_mean_m','reference_end_m']+(['model_mean_m','model_end_m','improvement_m'] if model else [])
    for (kind,key),rows in sorted(group.items()):
        outrow={'group':kind,'value':key,'count':len(rows)}
        for c in cols:outrow[c]=float(np.mean([r[c] for r in rows]))
        summary.append(outrow)
    with (out/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    print(f"Scenes={len(records)}. Saved {out/'per_scene.csv'} and {out/'summary.csv'}")
    for r in summary:
        if r['group'] in ('ALL','DISTANCE','VISIBILITY'):
            print(f"{r['group']:11s} {r['value']:>8s} n={r['count']:4d} reference={r['reference_mean_m']:.3f}m" + (f" model={r['model_mean_m']:.3f}m delta={r['improvement_m']:+.3f}m" if model else ''))

if __name__=='__main__':main()

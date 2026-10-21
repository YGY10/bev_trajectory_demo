"""Evaluate pretrained V4 model on NEW synthetic multi-lane intersection scenarios.
Run from bev_trajectory_demo root. Uses same Dataset/Model as V4.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import BEVDataset
from models import TrajectoryModel


def plot(rows,preds,paths):
    fig,axes=plt.subplots(3,3,figsize=(15,14));axes=axes.ravel()
    for ax,row,pred,ref in zip(axes,rows,preds,paths):
        for e in row['road_boundaries']:
            pts=np.array(e['points']);ax.plot(pts[:,0],pts[:,1],c='#50565b',lw=1.1,alpha=.8)
        for e in row['lane_markings']:
            pts=np.array(e['points']);ax.plot(pts[:,0],pts[:,1],c='#c5ab65' if e.get('color')=='yellow' else '#90a6b8',lw=1.,alpha=.85)
        gt=np.array(row['gt_path'])
        ax.plot(ref[:,0],ref[:,1],color='#e3a034',ls='--',lw=1.5,label='Reference')
        ax.plot(gt[:,0],gt[:,1],c='green',lw=2,label='Synthetic GT')
        ax.plot(pred[:,0],pred[:,1],c='red',lw=2,label='V4 zero-shot')
        ax.scatter([0],[0],c='blue',marker='^',s=30)
        ae=np.linalg.norm(pred-gt,axis=1)
        ax.set_title(f"{row['id']} | mean={ae.mean():.2f}m end={ae[-1]:.2f}m",fontsize=9)
        ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
    for ax in axes[len(rows):]:ax.axis('off')
    axes[0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(paths_out,dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default='data/intersection_v5_zeroshot.jsonl')
    parser.add_argument('--checkpoint',default='checkpoints/best_path_v4.pt')
    parser.add_argument('--output-dir',default='outputs/intersection_v5_zeroshot')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=parser.parse_args()
    global paths_out
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    paths_out=out/'prediction_grid.png'
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if ckpt.get('model_type') != 'arc_length_path_v4':
        raise ValueError('Need V4 PATH checkpoint (model_type=arc_length_path_v4)')
    dataset=BEVDataset(args.data,split='all',augment=False,ref_points=ckpt['horizon'])
    model=TrajectoryModel(horizon=ckpt['horizon']).to(args.device)
    model.load_state_dict(ckpt['model']);model.eval()
    data=[];preds=[];refs=[];result=[]
    for i,row in enumerate(dataset.rows):
        batch={k:v.unsqueeze(0).to(args.device) for k,v in dataset[i].items()}
        with torch.inference_mode():pred=model(batch)[0].cpu().numpy()
        gt=batch['target_xy'][0].cpu().numpy()
        ref=batch['reference'][0].cpu().numpy()
        err=np.linalg.norm(pred-gt,axis=1)
        # Exit accuracy: distance in final 20m; includes outbound-lane target deviation.
        end_mean=float(err[-20:].mean())
        result.append({'id':row['id'],'navigation':row['navigation'],
           'entry_lane':row['metadata']['entry_lane'],'exit_lane':row['metadata']['exit_lane'],
           'path_mean_m':float(err.mean()),'endpoint_m':float(err[-1]),
           'last_20m_mean_m':end_mean,
           'exit_near_lane_center':bool(end_mean < row['metadata']['lane_width_m']*.5)})
        preds.append(pred);refs.append(ref)
    with (out/'per_scene.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    with (out/'metrics.json').open('w') as f:json.dump(result,f,indent=2)
    plot(dataset.rows,preds,refs)
    for nav in ('straight','left','right'):
        sub=[r for r in result if r['navigation']==nav]
        if sub:
            print(f'{nav:<8} n={len(sub)} mean={np.mean([x["path_mean_m"] for x in sub]):.3f}m '
                  f'end={np.mean([x["endpoint_m"] for x in sub]):.3f}m '
                  f'exit-near-center={sum(x["exit_near_lane_center"] for x in sub)}/{len(sub)}')
    print(f'ALL mean={np.mean([x["path_mean_m"] for x in result]):.3f}m '
          f'end={np.mean([x["endpoint_m"] for x in result]):.3f}m')
    print(f'Saved: {out / "per_scene.csv"}, {paths_out}')
    print('Note: test set is synthetic. No human-likeness claim, no intersection traffic rules modeled.')
if __name__=='__main__':main()

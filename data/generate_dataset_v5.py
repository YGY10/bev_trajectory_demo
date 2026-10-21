#!/usr/bin/env python3
"""V5 intersection path dataset: geometry-consistent four-pose observations.
Only numpy/matplotlib required. Coordinates: local ego x forward, y left.
GT is synthetic geometry, NOT demonstrated human driving behaviour.
"""
import argparse, json, math, csv
from collections import Counter
from pathlib import Path
import numpy as np

NAVS=('straight','left','right')
DISTANCES=(50.,35.,20.,7.)

def arc_resample(points, spacing=1., n=81):
    p=np.asarray(points,dtype=float)
    lengths=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    keep=np.r_[True,np.diff(lengths)>1e-7]
    p,lengths=p[keep],lengths[keep]
    if lengths[-1] < (n-1)*spacing-1e-5: raise ValueError('global route too short')
    s=np.arange(n)*spacing
    return np.stack([np.interp(s,lengths,p[:,j]) for j in range(2)],axis=1)

def route_from_pose(points, pose_xy, count=81):
    p=np.asarray(points,float)
    d=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    # ego is on route on incoming straight (x=-distance, y=ego lane center 0)
    # The incoming straight begins at global x=-100; ego is exactly on it.
    # Use continuous arc position, not nearest sampled vertex (which offsets the start).
    start=float(pose_xy[0]-p[0,0])
    s=start+np.arange(count,dtype=float)
    if s[-1]>d[-1]: raise ValueError('route insufficient length')
    return np.column_stack([np.interp(s,d,p[:,j]) for j in (0,1)])

def bezier(a,b,c,d,n=320):
    t=np.linspace(0,1,n)[:,None]
    return (1-t)**3*a+3*(1-t)**2*t*b+3*(1-t)*t*t*c+t**3*d

def make_base(idx,seed):
    rng=np.random.default_rng(seed)
    nav=NAVS[idx%3]
    n_in=int(rng.integers(2,5));n_out=int(rng.integers(2,5))
    entry_idx=int(rng.integers(0,n_in));exit_idx=int(rng.integers(0,n_out))
    width_in=float(rng.uniform(3.15,3.75));width_out=float(rng.uniform(3.15,3.75))
    theta= {'straight':0.,'left':math.pi/2,'right':-math.pi/2}[nav]
    theta+=float(rng.uniform(-0.15,0.15))
    v=np.array([np.cos(theta),np.sin(theta)])
    norm=np.array([-v[1],v[0]])
    if nav=='straight':
        b=np.array([float(rng.uniform(31,43)),float(rng.uniform(-5,5))]);a=np.array([float(rng.uniform(3,8)),0.])
    else:
        b=np.array([float(rng.uniform(30,43)),float((1 if nav=='left' else -1)*rng.uniform(25,35))]);a=np.array([float(rng.uniform(3,8)),0.])
    run=float(np.linalg.norm(b-a));strength=float(rng.uniform(0.28,0.44))*run
    turn=bezier(a,a+np.array([strength,0.]),b-v*strength,b)
    first=np.column_stack((np.linspace(-100,a[0],1100),np.zeros(1100)))
    tail=b+np.linspace(0,160,900)[:,None]*v
    whole=np.vstack((first[:-1],turn[:-1],tail))
    return dict(id=f'junction_{idx:05d}',nav=nav,n_in=n_in,n_out=n_out,entry_idx=entry_idx,exit_idx=exit_idx,
                width_in=width_in,width_out=width_out,theta=theta,b=b,a=a,v=v,norm=norm,
                route=whole,seed=seed)

def clip_visible(poly,ego,forward=68.,rear=18.,side=50.,radius=69.,fov_deg=72.):
    """Clip polyline at true visibility boundaries via dense segment sampling.
    Multiple connected runs are retained rather than jumping over gaps.
    This synthetic distance/FOV mask is not a camera occlusion model.
    """
    p=np.asarray(poly,float)-ego
    out=[];run=[]
    # resample continuous input geometry to <=0.35m steps
    for a,b in zip(p[:-1],p[1:]):
        size=max(2,int(np.linalg.norm(b-a)/.35)+2)
        for t in np.linspace(0,1,size,endpoint=False):
            q=a*(1-t)+b*t
            x,y=q
            r=math.hypot(x,y)
            ok=(-rear<=x<=forward and abs(y)<=side and r<=radius and
                (x>=0 and abs(math.degrees(math.atan2(y,max(x,1e-6)))) <= fov_deg or x<0))
            if ok:run.append(q)
            elif run:
                if len(run)>=3 and np.linalg.norm(run[-1]-run[0])>=2:out.append(np.asarray(run))
                run=[]
    if run and len(run)>=3 and np.linalg.norm(run[-1]-run[0])>=2:out.append(np.asarray(run))
    return out

def make_observation(base,distance,pose_index,seed):
    rng=np.random.default_rng(seed)
    ego=np.array([-distance,0.]);nav=base['nav'];b=base['b'];v=base['v'];norm=base['norm']
    markings=[];boundaries=[]
    def append_line(pts,part,kind,color=None,style=None):
        # Each observed segment is a polyline; occasionally introduce a sensing gap.
        segments=clip_visible(pts,ego)
        for seg in segments:
            chunks=[seg]
            if kind=='lane_marking' and len(seg)>38 and rng.random()<0.18:
                k=int(rng.integers(15,len(seg)-15));gap=int(rng.integers(5,11))
                chunks=[seg[:k],seg[k+gap:]]
            for chunk in chunks:
                if len(chunk)<3 or np.linalg.norm(chunk[-1]-chunk[0])<2.:continue
                sigma=0.025 if kind=='road_boundary' else 0.04
                noise=rng.normal(0,sigma,size=chunk.shape)
                if len(chunk)>3:
                    noise=np.column_stack([np.convolve(noise[:,j],np.ones(3)/3,'same') for j in (0,1)])
                new=chunk+noise
                item={'type':kind,'points':np.round(new[::max(1,len(new)//100)],3).tolist(),
                      'confidence':round(float(rng.uniform(.83,.98) if kind=='road_boundary' else rng.uniform(.72,.97)),3),
                      'source_road':part}
                if kind=='lane_marking': item.update(color=color,line_style=style)
                (boundaries if kind=='road_boundary' else markings).append(item)
    # Start/end break creates an unmarked junction interior.
    entry_stop=float(base.get('entry_stop',-2.0))
    for j in range(base['n_in']+1):
        lateral=(j-.5-base['entry_idx'])*base['width_in']
        p=np.column_stack((np.linspace(-105,entry_stop,360),np.full(360,lateral)))
        is_bd=j in (0,base['n_in'])
        append_line(p,'entry','road_boundary' if is_bd else 'lane_marking',
                    color='yellow' if j==1 else 'white',style='dashed' if j%2 else 'solid')
    for j in range(base['n_out']+1):
        lateral=(j-.5-base['exit_idx'])*base['width_out']
        d=np.linspace(3,125,440)
        p=b[None,:]+d[:,None]*v[None,:]+lateral*norm[None,:]
        is_bd=j in (0,base['n_out'])
        append_line(p,'exit','road_boundary' if is_bd else 'lane_marking',
                    color='yellow' if j==1 else 'white',style='dashed' if j%2 else 'solid')
    # GT from one global route, re-centered on the ego
    gt=route_from_pose(base['route'],ego)-ego
    # Independent lane-grade reference; mid-intersection warp disappears near ego and far exit.
    extended=route_from_pose(base['route'],ego,count=101)-ego
    tang=np.gradient(extended,axis=0)
    tang/=np.maximum(np.linalg.norm(tang,axis=1,keepdims=True),1e-6)
    lateral=np.stack((-tang[:,1],tang[:,0]),axis=-1)
    s=np.arange(101,dtype=float)
    # Small global smooth bias plus a local deformation around corner; designed to maintain exit intent.
    amp=float(rng.uniform(.25,.65))*rng.choice([-1,1])
    transition=(np.sin(np.pi*s/100.)**2)
    corner_progress=distance+float(base['a'][0])+max(5.,np.linalg.norm(base['b']-base['a'])*.45)
    warp=amp*transition + rng.choice([-1,1])*float(rng.uniform(1.0,2.2))*np.exp(-.5*((s-corner_progress)/10.)**2)
    # Avoid losing target lane identity at visible exit far from the junction
    warp=np.clip(warp,-2.35,2.35)
    ref=extended+warp[:,None]*lateral
    ref-=ref[0]
    # The path starts exactly at the ego origin; reference has 101 points to avoid end interpolation artifacts.
    data={'id':f"{base['id']}_{nav}_pose{pose_index}_d{int(distance):02d}",
          'base_intersection_id':base['id'],'coordinate_system':'ego_at_start_x_forward_y_left_m',
          'navigation':nav, 'reference_line':np.round(ref,3).tolist(),
          'lane_markings':markings,'road_boundaries':boundaries,
          'gt_path':np.round(gt,3).tolist(),'gt_path_sample_spacing_m':1.,
          'metadata':{'synthetic':True,'generator_version':'v5_multiego_1',
                      'base_intersection_id':base['id'],'ego_distance_to_junction_m':distance,
                      'ego_pose_index':pose_index,'entry_lane_index':base['entry_idx'],
                      'exit_lane_index':base['exit_idx'],
                      'entry_lane_count':base['n_in'],'exit_lane_count':base['n_out'],
                      'lane_width_entry_m':base['width_in'],'lane_width_exit_m':base['width_out'],
                      'outgoing_heading_rad':base['theta'],
                      'target_lane_explicit_in_reference':True,
                      'intersection_no_lane_markings':True,
                      'observed_exit_lines':sum(e['source_road']=='exit' for e in markings+boundaries),
                      'observed_entry_lines':sum(e['source_road']=='entry' for e in markings+boundaries),
                      'fov_model':'ego-distance-and-angle geometric mask, no occlusion'}}
    return data

def make_rows(n_bases,seed):
    rows=[]
    for i in range(n_bases):
        base=make_base(i,seed+991*i)
        for j,d in enumerate(DISTANCES):
            rows.append(make_observation(base,d,j,seed+991*i+100+j))
    return rows

def audit(rows,max_lines=16):
    over=[];bad=[];stats={};counts=Counter()
    for r in rows:
        n=len(r['lane_markings'])+len(r['road_boundaries']);counts[r['navigation']]+=1
        if n>max_lines:over.append((r['id'],n))
        p=np.asarray(r['gt_path']); d=np.linalg.norm(np.diff(p,axis=0),axis=1)
        if len(p)!=81 or p.shape[1]!=2 or np.max(abs(d-1))>.025 or np.linalg.norm(p[0])>.025:
            bad.append((r['id'],len(p),float(np.max(abs(d-1)))))
    for nav in NAVS:
        rr=[r for r in rows if r['navigation']==nav]
        for pose in range(4):
            subset=[r for r in rr if r['metadata']['ego_pose_index']==pose]
            stats[(nav,pose)]=round(float(np.mean([r['metadata']['observed_exit_lines'] for r in subset])),2) if subset else None
    return over,bad,stats

def plot_preview(rows,path,bases=3):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(bases,4,figsize=(16,4*bases),squeeze=False)
    for i in range(bases):
        for j in range(4):
            ax=axes[i,j];r=rows[i*4+j]
            for ln in r['road_boundaries']:
                p=np.asarray(ln['points']);ax.plot(p[:,0],p[:,1],color='#777',lw=1)
            for ln in r['lane_markings']:
                p=np.asarray(ln['points']);ax.plot(p[:,0],p[:,1],color='#d5af56' if ln['color']=='yellow' else '#91b7cf',lw=1)
            gt=np.asarray(r['gt_path']);ref=np.asarray(r['reference_line'])
            ax.plot(gt[:,0],gt[:,1],color='green',lw=2,label='GT')
            ax.plot(ref[:,0],ref[:,1],color='darkorange',ls='--',lw=1.3,label='Reference')
            ax.scatter([0],[0],c='royalblue',marker='^',s=35)
            ax.set_xlim(-20,82);ax.set_ylim(-65,65);ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15)
            ax.set_title(f"{r['navigation']} | {int(DISTANCES[j])}m | exit={r['metadata']['observed_exit_lines']}",fontsize=9)
    axes[0,0].legend(fontsize=7);fig.tight_layout()
    Path(path).parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=140);plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bases',type=int,default=30)
    p.add_argument('--seed',type=int,default=20261021)
    p.add_argument('--output',default='data/synthetic_bev_paths_v5_preview.jsonl')
    p.add_argument('--preview',default='outputs/v5_preview/multi_ego.png')
    p.add_argument('--report',default='outputs/v5_preview/audit.csv')
    p.add_argument('--max-lines',type=int,default=16)
    args=p.parse_args()
    rows=make_rows(args.bases,args.seed)
    over,bad,stats=audit(rows,args.max_lines)
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('w',encoding='utf8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
    report=Path(args.report);report.parent.mkdir(parents=True,exist_ok=True)
    with report.open('w',newline='',encoding='utf8') as f:
        wr=csv.writer(f);wr.writerow(['id','nav','distance_m','line_count','observed_exit_lines','observed_entry_lines','over_limit'])
        for r in rows:
            meta=r['metadata'];n=len(r['lane_markings'])+len(r['road_boundaries'])
            wr.writerow([r['id'],r['navigation'],meta['ego_distance_to_junction_m'],n,meta['observed_exit_lines'],meta['observed_entry_lines'],int(n>args.max_lines)])
    plot_preview(rows,args.preview,min(3,args.bases))
    print(f'Generated {len(rows)} samples ({args.bases} bases × 4 poses); output={dest}')
    print('Polyline over capacity:',len(over),'path spacing errors:',len(bad))
    for nav in NAVS:print(nav,'exit visible lines at [50,35,20,7]m:',[stats[(nav,j)] for j in range(4)])
    print('Audit:',report,'Preview:',args.preview)
    print('Note: synthetic GT is not a human-driving demonstration.')
    if bad:raise SystemExit('Invalid GT path geometry')
    if over:raise SystemExit('Polyline capacity exceeded; inspect audit before training')
if __name__=='__main__':main()

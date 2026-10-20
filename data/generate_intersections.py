"""Small multi-lane intersection zero-shot test set for V4 time-free PathModel.
No model weights required. Purely synthetic geometry, NOT human driving evidence.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def arc_resample(points, n=81, spacing=1.0):
    p = np.asarray(points, dtype=float)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    keep = np.r_[True, np.diff(d) > 1e-8]
    p, d = p[keep], d[keep]
    if d[-1] < (n - 1) * spacing:
        raise ValueError(f'Geometry too short: {d[-1]:.1f}m, need >= {(n-1)*spacing:.1f}m')
    t = np.arange(n) * spacing
    return np.stack([np.interp(t, d, p[:, i]) for i in range(2)], axis=-1)


def bezier(a,b,c,d,n=200):
    t = np.linspace(0,1,n)[:,None]
    return (1-t)**3*a + 3*(1-t)**2*t*b + 3*(1-t)*t*t*c + t**3*d


def clip_polyline(points, bounds):
    """Clip segments against axis-aligned BEV box, maintaining disconnected runs."""
    xmin,xmax,ymin,ymax = bounds
    p = np.asarray(points, float)
    runs=[]; cur=[]
    def cut(a,b):
        delta=b-a; lo=0.; hi=1.
        for u,v in ((-delta[0],a[0]-xmin),(delta[0],xmax-a[0]),(-delta[1],a[1]-ymin),(delta[1],ymax-a[1])):
            if abs(u)<1e-12:
                if v<0: return None
            else:
                t=v/u
                if u<0:lo=max(lo,t)
                else:hi=min(hi,t)
                if lo>hi:return None
        return a+lo*delta,a+hi*delta
    for a,b in zip(p[:-1],p[1:]):
        segment=cut(a,b)
        if segment is None:
            if len(cur)>1:runs.append(np.asarray(cur))
            cur=[];continue
        u,v=segment
        if cur and np.linalg.norm(u-cur[-1])>1e-4:
            if len(cur)>1:runs.append(np.asarray(cur))
            cur=[]
        if not cur:cur=[u]
        if np.linalg.norm(v-cur[-1])>1e-5:cur.append(v)
    if len(cur)>1:runs.append(np.asarray(cur))
    return runs


def polyline_samples(a, b, step=.75):
    n=max(3,int(np.linalg.norm(b-a)/step)+1)
    return a[None,:]+np.linspace(0,1,n)[:,None]*(b-a)[None,:]


def create_scene(nav, entry_idx, exit_idx, variant, seed):
    rng=np.random.default_rng(seed)
    w=float(rng.uniform(3.3,3.75))
    # Three entry lanes, ego at selected lane center. Outbound lanes are independent.
    side=(np.array([0.,1.]))
    def entry_center(i): return np.array([0.,(i-entry_idx)*w])
    direction={'straight':0.,'left':np.pi/2,'right':-np.pi/2}[nav]
    direction+=float(rng.uniform(-.07,.07))
    v=np.array([np.cos(direction),np.sin(direction)])
    normal=np.array([-v[1],v[0]])
    if nav=='straight':
        exit_anchor=np.array([float(rng.uniform(48,57)),float(rng.uniform(-5.0,5.0))])
    else:
        exit_anchor=np.array([float(rng.uniform(37,48)),float(np.sign(direction)*rng.uniform(18,24))])
    # Anchor is the TARGET lane center (E if exit_idx==1), not the roadway center.
    # Physical road coordinate system shifts according to target exit lane.
    start_curve=float(rng.uniform(15,22))
    a=np.array([start_curve,0.])
    b=exit_anchor.copy()
    distance=float(np.linalg.norm(b-a))
    strength=float(rng.uniform(.26,.41))*distance
    turn=bezier(a,a+np.array([strength,0.]),b-v*strength,b,n=260)
    entry=np.stack([np.linspace(0,start_curve,70),np.zeros(70)],axis=1)
    tail=b+np.linspace(0,120,240)[:,None]*v[None,:]
    geometry=np.vstack((entry[:-1],turn[:-1],tail))
    gt=arc_resample(geometry)
    # road lines end at intersection and restart at exit: NO internal lane markings
    stop_x=float(rng.uniform(9,13))
    entry_end=float(rng.uniform(18,22))
    leave_start=float(rng.uniform(1,4))
    markings=[];boundaries=[]
    xmin,xmax,ymin,ymax=((-80.,80.,-10.,10.) if nav=='straight' else (-60.,60.,-20.,20.))
    bounds=(xmin,xmax,ymin,ymax)
    def add_geom(source, kind, color=None, style=None, boundary=False):
        for segment in clip_polyline(source,bounds):
            if len(segment)<3:continue
            # light correlated perception noise and rare small missing segment
            sigma=0.035 if not boundary else 0.018
            noise=rng.normal(0,sigma,size=segment.shape)
            if len(segment)>4:
                noise=np.stack([np.convolve(noise[:,j],np.ones(3)/3,'same') for j in range(2)],axis=1)
            segment=segment+noise
            pieces=[segment]
            if not boundary and len(segment)>20 and rng.random()<.22:
                pos=int(rng.integers(7,len(segment)-6))
                pieces=[segment[:pos],segment[pos+int(rng.integers(2,5)):]]
            for part in pieces:
                if len(part)<3:continue
                obj={'type':'road_boundary' if boundary else 'lane_marking',
                     'points':np.round(part,3).tolist(),
                     'confidence':round(float(rng.uniform(.86,.98) if boundary else rng.uniform(.72,.97)),3)}
                if not boundary:obj.update(color=color,line_style=style)
                (boundaries if boundary else markings).append(obj)
    # Entry roadway lane centers are [A,B,C] relative to ego lane.
    for j in range(4):
        off=(j-.5-entry_idx)*w
        e=polyline_samples(np.array([-28.,off]),np.array([entry_end,off]))
        if j in (0,3):add_geom(e,'boundary',boundary=True)
        else:add_geom(e,'marking',color='yellow' if j==1 else 'white',style='dashed' if j==2 else 'solid')
    # Turn exit is visible through a short, direction-aligned BEV corridor.
    # A single ego-aligned y=+-20 m rectangle would delete ALL outgoing
    # lines, as these begin at y~20-24 m. This intentionally models an
    # additional turn-facing perception sector, not a rigid 20m lateral FOV.
    exit_seen_m = float(rng.uniform(22., 35.)) if nav != 'straight' else 140.
    for j in range(4):
        off=(j-.5-exit_idx)*w
        start=b+v*leave_start+normal*off
        end=b+v*(exit_seen_m if nav!='straight' else 140.)+normal*off
        e=polyline_samples(start,end)
        if nav != 'straight':
            # Crop by an expanded global envelope; local exit length is
            # already bounded above. No artificial continuation is added.
            original_bounds = bounds
            bounds = (-60.,60.,-65.,65.)
        if j in (0,3):add_geom(e,'boundary',boundary=True)
        else:add_geom(e,'marking',color='yellow' if j==1 else 'white',style='dashed' if j==2 else 'solid')
        if nav != 'straight':bounds = original_bounds
    # Lane-grade biased reference, same topology and destination; can be imperfect in unmarked junction.
    s=np.arange(100,dtype=float)
    base=arc_resample(geometry,100,1.)
    tangent=np.gradient(base,axis=0)
    tangent=tangent/np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-8)
    nrm=np.stack([-tangent[:,1],tangent[:,0]],axis=-1)
    # Smooth lateral warp <=0.85m so it cannot jump to adjacent 3.3+m lane.
    amplitude=float(rng.uniform(.35,.82))
    warp=amplitude*np.sin(np.pi*s/99)**2*np.sin((s-15)/18+float(rng.uniform(-1,1)))
    # Extra deformation in unmarked intersection; still within target lane catchment.
    warp+=float(rng.choice([-1,1])*rng.uniform(1.6,2.8))*np.exp(-((s-36)/10)**2)
    reference=base+warp[:,None]*nrm
    # Keep reference start aligned with ego. Endpoint stays near selected outbound lane.
    reference-=reference[:1]
    return {'id':f'intersection_{nav}_{"ABC"[entry_idx]}2{"DEF"[exit_idx]}_{variant:02d}',
      'coordinate_system':'ego_at_start_x_forward_y_left_m','navigation':nav,
      'reference_line':np.round(reference,3).tolist(),
      'lane_markings':markings,'road_boundaries':boundaries,
      'gt_path':np.round(gt,3).tolist(),'gt_path_sample_spacing_m':1.0,
      'metadata':{'synthetic':True,'generator_version':'v5_intersection_zero_shot',
      'entry_lane':'ABC'[entry_idx],'exit_lane':'DEF'[exit_idx],
      'target_lane_explicit_in_reference':True,'intersection_no_lane_markings':True,
      'lane_width_m':round(w,3),'intersection_gap_start_x_m':round(entry_end,2),
      'outgoing_heading_rad':round(direction,5),'bev_box_m':[xmin,xmax,ymin,ymax],
      'turn_exit_corridor_length_m':round(exit_seen_m,2) if nav != 'straight' else None,
      'bev_shape_note':'turn BEV includes an exit-oriented visibility corridor outside the ego-axis rectangular core' if nav != 'straight' else 'ego-axis rectangular FOV',
      'path_length_m':80,'reference_warp_max_m':round(float(np.max(np.abs(warp))),3)}}


def plot_scenes(rows, outfile, n=9):
    k=min(len(rows),n);fig,axes=plt.subplots(3,3,figsize=(15,14));axes=axes.ravel()
    for ax,row in zip(axes,rows[:k]):
        for line in row['road_boundaries']:
            p=np.array(line['points']);ax.plot(p[:,0],p[:,1],color='#444444',lw=1.0)
        for line in row['lane_markings']:
            p=np.array(line['points']);ax.plot(p[:,0],p[:,1],color='#c4a65b' if line.get('color')=='yellow' else '#9caec4',lw=.9)
        p=np.array(row['gt_path']);r=np.array(row['reference_line'])
        ax.plot(r[:,0],r[:,1],color='orange',ls='--',lw=1.8,label='Biased reference')
        ax.plot(p[:,0],p[:,1],color='green',lw=2.5,label='GT path')
        ax.scatter([0],[0],marker='^',c='blue',s=35)
        ax.set_title(row['id'],fontsize=10);ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
    for ax in axes[k:]:ax.axis('off')
    axes[0].legend(fontsize=8);fig.tight_layout();Path(outfile).parent.mkdir(parents=True,exist_ok=True);fig.savefig(outfile,dpi=140);plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='data/intersection_v5_zeroshot.jsonl')
    p.add_argument('--preview',default='outputs/intersection_v5_preview.png')
    p.add_argument('--seed',type=int,default=20261018)
    p.add_argument('--variants',type=int,default=3,help='Variants per direction; three lane pairs cycle across variants')
    args=p.parse_args()
    rows=[]
    for nav_index,nav in enumerate(('straight','left','right')):
        for idx in range(args.variants):
            # For example 0: B -> E; other variants test differing lane selections.
            entry,exit_ = [(1,1),(0,2),(2,0)][idx%3]
            rows.append(create_scene(nav,entry,exit_,idx,args.seed+100*nav_index+idx))
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    plot_scenes(rows,args.preview)
    print(f'Saved {len(rows)} scenes: {out}; preview: {args.preview}')
    print('IMPORTANT: test-only intersection dataset; do not treat these as real human path labels')

if __name__=='__main__':main()

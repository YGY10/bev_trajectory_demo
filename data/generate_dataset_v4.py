#!/usr/bin/env python3
"""Synthetic 80m spatial PATH dataset V4.

Polyline geometry is clipped to probabilistic BEV FOV, then moderately corrupted
with correlated localization noise and occasional contiguous visibility gaps.
GT path and reference are generated separately and are NOT corrupted by BEV noise.
Ego frame: x forward, y left. No time dimension in gt_path.
"""
import argparse
import json
from pathlib import Path
import numpy as np

COMMANDS = ('straight', 'left', 'right')


def smoothstep(x):
    x = np.clip(x, 0., 1.)
    return x*x*(3-2*x)


def arc_resample(points, distances):
    p = np.asarray(points, dtype=np.float64)
    d = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    ok = np.r_[True, np.diff(d)>1e-8]
    p, d = p[ok], d[ok]
    if distances[-1] > d[-1]+1e-5:
        raise ValueError('Path not long enough for requested arc-length horizon')
    return np.column_stack([np.interp(distances, d, p[:,k]) for k in range(2)])


def geometry(nav, rng):
    ds = .12
    s = np.arange(0., 140.+ds*.5, ds)
    turn_start = float(rng.uniform(13, 32))
    turn_span = float(rng.uniform(13, 33))
    direction = {'straight': 0, 'left': 1, 'right': -1}[nav]
    theta = direction*(np.pi/2 + rng.uniform(-.055, .055))*smoothstep((s-turn_start)/turn_span)
    if nav == 'straight':
        theta = rng.uniform(-.045, .045)*smoothstep(s/rng.uniform(40, 80))
    theta += rng.uniform(-.007, .007)*np.sin(s/24)*smoothstep(s/16)
    theta_mid = .5*(theta[1:]+theta[:-1])
    centers = np.vstack((np.zeros((1,2)), np.cumsum(ds*np.column_stack((np.cos(theta_mid),np.sin(theta_mid))),axis=0)))
    normal = np.column_stack((-np.sin(theta),np.cos(theta)))
    return s, centers, normal, turn_start, turn_span


def reference(rng, s, centers, normal, start, span, path_length):
    # Warp longitudinal progress locally so the reference's turning position differs.
    distance = np.arange(0., path_length+25.1, 1.)
    shift = float(rng.uniform(-5, 5))
    scale = float(rng.uniform(.82, 1.22))
    region = smoothstep((distance-(start-13))/11)*(1-smoothstep((distance-(start+span+23))/15))
    shifted = np.clip(distance+region*((distance-start)*(1/scale-1)+shift),0,s[-1])
    xy = np.column_stack([np.interp(shifted,s,centers[:,j]) for j in (0,1)])
    nn = np.column_stack([np.interp(shifted,s,normal[:,j]) for j in (0,1)])
    nn /= np.maximum(np.linalg.norm(nn,axis=1,keepdims=True),1e-8)
    offset = float(rng.uniform(-2.5,2.5))
    drift = float(rng.uniform(-1.1,1.1))
    lateral = offset+drift*np.sin(distance/19+float(rng.uniform(-3,3)))
    xy += lateral[:,None]*nn
    # Map bias smooth, no independent point jitter.
    return xy.round(3).tolist(), {'lateral_bias_m':round(offset,3),'turn_shift_m':round(shift,3),'turn_scale':round(scale,3)}


def clip_one_segment(p0,p1,xlim,ylim):
    """Liang-Barsky clipping of one 2D segment to the ego-centered BEV rectangle."""
    x0,y0=p0; dx,dy=np.asarray(p1)-p0
    t0,t1=0.,1.
    for p,q in [(-dx,x0+xlim),(dx,xlim-x0),(-dy,y0+ylim),(dy,ylim-y0)]:
        if abs(p)<1e-12:
            if q<0: return None
        else:
            r=q/p
            if p<0: t0=max(t0,r)
            else: t1=min(t1,r)
            if t0>t1: return None
    return np.asarray(p0)+t0*np.array([dx,dy]),np.asarray(p0)+t1*np.array([dx,dy])


def clip_polyline(points,xlim,ylim):
    pieces=[]; current=[]
    def finish():
        nonlocal current
        if len(current)>1 and np.linalg.norm(np.asarray(current[-1])-current[0])>1e-5:
            pieces.append(np.asarray(current))
        current=[]
    for a,b in zip(points[:-1],points[1:]):
        c=clip_one_segment(a,b,xlim,ylim)
        if c is None:
            finish(); continue
        p,q=c
        if current and np.linalg.norm(np.asarray(current[-1])-p)>1e-4:
            finish()
        if not current: current=[p]
        if np.linalg.norm(current[-1]-q)>1e-7: current.append(q)
    finish()
    return pieces


def sample_fov(rng, nav):
    """High probability of reaching the nominal FOV, not a hard fixed limit.

    x is ego-forward; +- is used intentionally, matching the earlier V3 convention.
    """
    if nav == 'straight':
        xlim = rng.uniform(72., 88.) if rng.random() < .8 else rng.uniform(48., 72.)
        ylim = rng.uniform(9., 11.) if rng.random() < .8 else rng.uniform(7., 9.)
    else:
        xlim = rng.uniform(54., 66.) if rng.random() < .8 else rng.uniform(35., 54.)
        ylim = rng.uniform(18., 22.) if rng.random() < .8 else rng.uniform(13., 18.)
    return float(xlim), float(ylim)


def corrupt_piece(piece, rng, kind, severity):
    """Corrupt *visible* polyline, leaving GT and reference unchanged.

    Noise is spatially correlated (not independent point salt-and-pepper). Gaps
    remove contiguous subsegments; each surviving run remains point-ordered.
    """
    piece = np.asarray(piece, dtype=float)
    if len(piece) < 2:
        return []
    # Densify at ~1 m resolution first for smooth noise and meaningful gaps.
    dist = np.r_[0., np.cumsum(np.linalg.norm(np.diff(piece, axis=0), axis=1))]
    length = dist[-1]
    if length < 2.:
        return []
    ss = np.arange(0., length, 1.)
    ss = np.r_[ss, length] if not len(ss) or abs(ss[-1] - length) > .25 else ss
    xy = np.column_stack([np.interp(ss, dist, piece[:, j]) for j in (0, 1)])
    n = len(xy)
    if n < 3:
        return []
    if severity == 'none':
        scale = 0.
        p_gap = 0.
    else:
        level = {'mild': .65, 'medium': 1., 'strong': 1.5}[severity]
        scale = (0.11 if kind == 'lane_marking' else 0.055) * level
        p_gap = (0.55 if kind == 'lane_marking' else 0.24) * level
    if scale:
        # Smooth positional uncertainty; turn geometry remains structurally sound.
        knots = np.arange(0., length + 8., 8.)
        disp = rng.normal(0., scale, (len(knots), 2))
        smooth = np.column_stack([np.interp(ss, knots, disp[:, j]) for j in (0, 1)])
        jitter = rng.normal(0., scale * .22, (n, 2))
        xy += smooth + jitter
    keep = np.ones(n, dtype=bool)
    if rng.random() < p_gap and length > 12.:
        ngaps = 1 if rng.random() < .8 else 2
        for _ in range(ngaps):
            gap_length = rng.uniform(2., 5.5) if kind == 'lane_marking' else rng.uniform(1.5, 3.)
            a = rng.uniform(3., max(3.1, length - gap_length - 2.))
            keep &= ~((ss >= a) & (ss <= a + gap_length))
    # Return separate contiguous runs, reject very short fragments.
    runs = []
    start = None
    for j in range(n + 1):
        valid = j < n and bool(keep[j])
        if valid and start is None:
            start = j
        elif not valid and start is not None:
            part = xy[start:j]
            if len(part) >= 3 and np.linalg.norm(np.diff(part, axis=0), axis=1).sum() >= 2.5:
                runs.append(part)
            start = None
    return runs


def line_elements(rng, s, centers, normal, offset, kind, xlim, ylim,
                  severity, color=None, style=None):
    path = centers + offset * normal
    # Clip ideal line first; only simulate sensor error after sensing FOV clipping.
    sampled = path[::8]
    result = []
    for piece in clip_polyline(sampled, xlim, ylim):
        for visible in corrupt_piece(piece, rng, kind, severity):
            if rng.random() < .5:
                visible = visible[::-1]
            # Keep every point finite; intentionally permit tiny sensor error outside
            # ideal FOV near the border, as actual measurements can fluctuate.
            item = {'type': kind, 'points': visible.round(3).tolist(),
                    'confidence': round(float(rng.uniform(.77, .98) if kind == 'lane_marking'
                                             else rng.uniform(.88, .99)), 3)}
            if kind == 'lane_marking':
                item.update(color=color, line_style=style)
            result.append(item)
    return result


def make_scene(i,seed=20261018,path_length=80,spacing=1.,severity="medium"):
    rng=np.random.default_rng(np.random.SeedSequence([seed,i]))
    nav=COMMANDS[i%3]
    s,centers,normal,start,span=geometry(nav,rng)
    # Pseudo-human geometric path: small smooth within-lane bias independent from map corruption.
    style_bias=float(rng.uniform(-.15,.15))
    extra=(.25 if nav=='left' else -.25 if nav=='right' else 0.)*rng.uniform(.4,1.5)
    offset=style_bias+extra*np.exp(-.5*((s-(start+span*.6))/max(span*.42,1.))**2)
    full_path=centers+offset[:,None]*normal
    full_path-=full_path[0].copy()
    sample_s=np.arange(0,path_length+spacing*.5,spacing)
    gt=arc_resample(full_path,sample_s)
    gt[0]=0.
    ref,ref_meta=reference(rng,s,centers,normal,start,span,path_length)
    lane_w=float(rng.uniform(3.25,3.85))
    xlim, ylim = sample_fov(rng, nav)
    markings=[]; boundaries=[]
    for j,off in enumerate((-1.5*lane_w,-.5*lane_w,.5*lane_w,1.5*lane_w)):
        markings+=line_elements(rng,s,centers,normal,off,'lane_marking',xlim,ylim,severity,
                                color='yellow' if j==0 else 'white',style='solid' if j in (0,3) else 'dashed')
    for off in (-2*lane_w,2*lane_w):
        boundaries+=line_elements(rng,s,centers,normal,off,'road_boundary',xlim,ylim,severity)
    # Keep a stable budget for the currently deployed Dataset (max 16 lines).
    # Prefer longer, information-rich pieces over tiny fragmentation leftovers.
    def length(item):
        pts = np.asarray(item['points'])
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    boundaries = sorted(boundaries, key=length, reverse=True)[:4]
    markings = sorted(markings, key=length, reverse=True)[:max(0, 16 - len(boundaries))]
    rng.shuffle(markings); rng.shuffle(boundaries)
    assert len(markings) + len(boundaries) <= 16
    return {'id':f'scene_{i:05d}', 'coordinate_system':'ego_at_start_x_forward_y_left_m',
            'navigation':nav,
            'reference_line':ref,'lane_markings':markings,'road_boundaries':boundaries,
            'gt_path':gt.round(4).tolist(),
            'gt_path_sample_spacing_m':spacing,
            'metadata':{'synthetic':True,'generator_version':4,'path_length_m':path_length,
                        'bev_x_min_m':-xlim,'bev_x_max_m':xlim,'bev_y_min_m':-ylim,'bev_y_max_m':ylim,
                        'bev_visibility':'random_fov_correlated_noise_contiguous_gaps',
                        'bev_noise_severity':severity,
                        'lane_width_m':round(lane_w,3),'turn_start_m':round(start,3),
                        'turn_span_m':round(span,3),**ref_meta,
                        'gt_note':'procedural geometric pseudo-human path, no timing'}}


def preview(records,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,len(records),figsize=(16,5.5),constrained_layout=True)
    for ax,r in zip(axs,records):
        for e in r['lane_markings']:
            p=np.asarray(e['points']);ax.plot(p[:,0],p[:,1],color='#c7aa56' if e['color']=='yellow' else '#777',lw=1.2)
        for e in r['road_boundaries']:
            p=np.asarray(e['points']);ax.plot(p[:,0],p[:,1],color='#292929',lw=1.7)
        for key,c,ls in [('reference_line','#f49a00','--'),('gt_path','#158b39','-')]:
            p=np.asarray(r[key]);ax.plot(p[:,0],p[:,1],color=c,ls=ls,lw=2.2,label=key)
        ax.scatter([0],[0],marker='^',color='blue',s=32)
        m=r['metadata']; ax.set_title(f"{r['navigation']} | BEV x±{m['bev_x_max_m']:.0f}, y±{m['bev_y_max_m']:.0f} m")
        ax.axis('equal');ax.grid(alpha=.2);ax.legend(fontsize=8)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=145);plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--count',type=int,default=3000)
    ap.add_argument('--bev-noise',choices=['none','mild','medium','strong'],default='medium')
    ap.add_argument('--seed',type=int,default=20261018)
    ap.add_argument('--path-length',type=int,default=80)
    ap.add_argument('--spacing',type=float,default=1.)
    ap.add_argument('--output',default='data/synthetic_bev_paths_v4.jsonl')
    ap.add_argument('--preview',default='data/sample_scenes_v4.png')
    args=ap.parse_args()
    if args.count<3 or args.path_length<=0 or args.spacing<=0 or abs(args.path_length/args.spacing-round(args.path_length/args.spacing))>1e-7:
        ap.error('count>=3 and path-length/spacing must be a positive integer')
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    samples=[]
    with output.open('w') as fp:
        for i in range(args.count):
            r=make_scene(i,args.seed,args.path_length,args.spacing,args.bev_noise)
            fp.write(json.dumps(r,separators=(',',':'))+'\n')
            if i<3:samples.append(r)
    if args.preview:preview(samples,args.preview)
    print(f'Wrote {args.count} scenes to {output}; {len(samples[0]["gt_path"])} points per path.')
    print('Nominal high-probability FOV: straight x=+-80 y=+-10; turn x=+-60 y=+-20.')
    print(f'BEV sensor corruption level: {args.bev_noise}')
    print('NOTE: Old trajectory training loader is NOT compatible with gt_path labels.')

if __name__=='__main__':main()

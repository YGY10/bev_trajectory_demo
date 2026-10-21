"""Split V5 by base intersection ID, never by ego observation."""
import argparse, json, random
from collections import Counter, defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--out-dir', default='data/v5_splits')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train-frac', type=float, default=.8)
    p.add_argument('--val-frac', type=float, default=.1)
    args = p.parse_args()
    if args.train_frac <= 0 or args.val_frac <= 0 or args.train_frac + args.val_frac >= 1:
        p.error('Require 0 < train-frac, val-frac and train-frac + val-frac < 1')
    rows=[json.loads(line) for line in open(args.data, encoding='utf-8') if line.strip()]
    groups=defaultdict(list)
    for row in rows:
        m=row.get('metadata',{})
        bid=row.get('base_intersection_id',m.get('base_intersection_id'))
        if not bid: raise ValueError('Missing base_intersection_id in scene '+str(row.get('id')))
        groups[str(bid)].append(row)
    by_nav=defaultdict(list)
    for bid, items in groups.items():
        navs={r['navigation'] for r in items}
        if len(navs)!=1: raise ValueError(f'Inconsistent nav within base {bid}: {navs}')
        by_nav[next(iter(navs))].append(bid)
    rng=random.Random(args.seed)
    assignment={}
    for nav, ids in sorted(by_nav.items()):
        rng.shuffle(ids)
        n=len(ids)
        n_tr=round(n*args.train_frac)
        n_val=round(n*args.val_frac)
        if n>=3:
            n_tr=max(1,min(n_tr,n-2))
            n_val=max(1,min(n_val,n-n_tr-1))
        for i,bid in enumerate(ids):
            assignment[bid]=('train' if i<n_tr else 'val' if i<n_tr+n_val else 'test')
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    handles={s:(out/f'{s}.jsonl').open('w',encoding='utf-8') for s in ('train','val','test')}
    counts=Counter()
    try:
        for row in rows:
            bid=str(row.get('base_intersection_id', row.get('metadata',{}).get('base_intersection_id')))
            split=assignment[bid]
            handles[split].write(json.dumps(row,ensure_ascii=False)+'\n')
            counts[split]+=1
    finally:
        for fh in handles.values():fh.close()
    assert not (set(assignment)==set())
    manifest={'input':args.data,'seed':args.seed,'counts':dict(counts),'bases_by_split':{s:sum(v==s for v in assignment.values()) for s in ('train','val','test')}, 'assignments':assignment}
    (out/'split_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    for split in ('train','val','test'):
        bids={bid for bid,s in assignment.items() if s==split}
        print(f'{split:5s}: {counts[split]:5d} samples, {len(bids):4d} base intersections')
    print('Saved to:',out)

if __name__=='__main__':main()

"""V5 trainer with group-disjoint JSONL files; does not overwrite V4 checkpoints."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import BEVDataset
from models import TrajectoryModel
from train import path_loss


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-data',default='data/v5_splits/train.jsonl')
    ap.add_argument('--val-data',default='data/v5_splits/val.jsonl')
    ap.add_argument('--epochs',type=int,default=60)
    ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--lr',type=float,default=3e-4)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--output',default='checkpoints/best_path_v5.pt')
    ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--workers',type=int,default=0)
    args=ap.parse_args()
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    train=BEVDataset(args.train_data,split='all',augment=False, max_lines=32)
    train.augment=True  # preserve V4 line-order / point-direction augmentation; no re-splitting
    val=BEVDataset(args.val_data,split='all',augment=False)
    assert train.path_points==val.path_points and train.spacing==val.spacing
    train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=args.workers)
    val_loader=DataLoader(val,batch_size=args.batch_size,shuffle=False,num_workers=args.workers)
    model=TrajectoryModel(horizon=train.path_points).to(args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    best=float('inf')
    for epoch in range(1,args.epochs+1):
        model.train();total=0.
        for batch in train_loader:
            batch={k:v.to(args.device) for k,v in batch.items()}
            pred=model(batch)
            loss,_,_=path_loss(pred,batch['target_xy'],train.spacing)
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            total+=loss.item()*pred.shape[0]
        model.eval();dist=[];seg=[]
        with torch.inference_mode():
            for batch in val_loader:
                batch={k:v.to(args.device) for k,v in batch.items()}
                pred=model(batch)
                dist.append(torch.linalg.vector_norm(pred-batch['target_xy'],dim=-1).cpu())
                seg.append((torch.linalg.vector_norm(pred[:,1:]-pred[:,:-1],dim=-1)-train.spacing).abs().cpu())
        d=torch.cat(dist)
        mean,end,spacing=d.mean().item(),d[:,-1].mean().item(),torch.cat(seg).mean().item()
        print(f'Epoch {epoch:03d}: train_loss={total/len(train):.4f} val_path_mean={mean:.3f}m val_path_end={end:.3f}m val_spacing_err={spacing:.3f}m',flush=True)
        if mean<best:
            best=mean
            Path(args.output).parent.mkdir(parents=True,exist_ok=True)
            torch.save({'model':model.state_dict(),'model_type':'arc_length_path_v5','horizon':train.path_points,'spacing':train.spacing,'epoch':epoch,'seed':args.seed,'best_val_path_mean':best},args.output)
    print('Saved best checkpoint:',args.output)

if __name__=='__main__':main()

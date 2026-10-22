#!/usr/bin/env python3
"""Diagnose raw and effective path residuals of V6B1 point-token checkpoints.

Run at the bev_trajectory_demo repository root. Does not modify weights or train.
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import BEVDataset
from train_v5_exit_lane_loss_v2 import build_model, MODEL_INPUT_KEYS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', nargs='+', default=[
        'checkpoints/v6B1_cnn_point_60ep_best_ade.pt',
        'checkpoints/v6B1_cnn_point_60ep_best_exit_lat.pt',
    ])
    p.add_argument('--data', default='data/v5_splits/val.jsonl')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--limit', type=int, default=300, help='0 means full dataset')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('--batch-size must be positive')

    for path in args.checkpoint:
        if not Path(path).exists():
            print(f'NOT FOUND: {path}')
            continue
        ck = torch.load(path, map_location='cpu', weights_only=True)
        variant = ck.get('model_variant', 'v6B1')
        encoder = ck.get('bev_encoder') or 'cnn'
        mode = ck.get('bev_token_mode') or ck.get('config', {}).get('bev_token_mode', 'pooled')
        horizon = int(ck['horizon'])
        model, _ = build_model(variant, horizon, encoder, mode)
        model.load_state_dict(ck['model'], strict=True)
        model = model.to(args.device).eval()

        ds = BEVDataset(args.data, split='all', augment=False,
                        max_lines=int(ck.get('max_lines', 32)), ref_points=horizon)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

        capture = {}
        def capture_head(_module, inputs, output):
            capture['decoded'] = inputs[0].detach()
            capture['raw'] = output.detach()
        handle = model.head.register_forward_hook(capture_head)

        totals = {k: 0.0 for k in (
            'raw_abs', 'raw_std_path', 'raw_centered_abs', 'effective_abs',
            'pred_ref_abs', 'pred_gt_abs', 'ref_gt_abs', 'decoded_std_path',
            'pred_ref_max_sum', 'raw_origin_abs', 'effective_std_path',
        )}
        total_scenes = 0
        total_points = 0
        max_pred_ref = 0.0
        first_example = None
        try:
            with torch.inference_mode():
                for batch in loader:
                    if args.limit > 0 and total_scenes >= args.limit:
                        break
                    remaining = args.limit-total_scenes if args.limit > 0 else batch['lines'].shape[0]
                    if remaining < batch['lines'].shape[0]:
                        batch = {k: v[:remaining] for k, v in batch.items()}
                    data = {k: batch[k].to(args.device) for k in MODEL_INPUT_KEYS}
                    pred = model(data)
                    raw = capture['raw']
                    decoded = capture['decoded']
                    ref = data['reference']
                    gt = batch['target_xy'].to(args.device)
                    effective = raw - raw[:, :1]
                    ref_relative = ref - ref[:, :1]
                    reconstructed = ref_relative + effective
                    if not torch.allclose(pred, reconstructed, rtol=1e-4, atol=2e-5):
                        raise AssertionError('Prediction != (ref-ref0)+(raw-raw0). Check model forward.')
                    count = pred.shape[0]
                    if first_example is None:
                        first_example = dict(
                            raw_first=raw[0,0].tolist(),
                            raw_mid=raw[0, horizon//2].tolist(),
                            raw_last=raw[0,-1].tolist(),
                            effective_last=effective[0,-1].tolist(),
                        )
                    totals['raw_abs'] += raw.abs().mean((1,2)).sum().item()
                    totals['raw_std_path'] += raw.std(dim=1, unbiased=False).norm(dim=-1).sum().item()
                    totals['raw_centered_abs'] += (raw-raw.mean(dim=1, keepdim=True)).norm(dim=-1).mean(dim=1).sum().item()
                    totals['raw_origin_abs'] += raw[:,0].norm(dim=-1).sum().item()
                    totals['effective_abs'] += effective.norm(dim=-1).mean(dim=1).sum().item()
                    totals['effective_std_path'] += effective.std(dim=1, unbiased=False).norm(dim=-1).sum().item()
                    totals['decoded_std_path'] += decoded.std(dim=1, unbiased=False).norm(dim=-1).sum().item()
                    totals['pred_ref_abs'] += (pred-ref_relative).norm(dim=-1).mean(dim=1).sum().item()
                    totals['pred_gt_abs'] += (pred-gt).norm(dim=-1).mean(dim=1).sum().item()
                    totals['ref_gt_abs'] += (ref_relative-gt).norm(dim=-1).mean(dim=1).sum().item()
                    max_pred_ref = max(max_pred_ref, (pred-ref_relative).norm(dim=-1).max().item())
                    total_scenes += count
                    total_points += count*horizon
        finally:
            handle.remove()
        if total_scenes == 0:
            raise RuntimeError('No scenes evaluated')
        print(f'\n===== {path} =====')
        print(f"checkpoint epoch: {ck.get('epoch')} model={variant} encoder={encoder} token_mode={mode}")
        print(f'scenes={total_scenes}, horizon={horizon}, device={args.device}')
        print('Units: meters unless stated otherwise')
        for k, v in totals.items():
            if k != 'pred_ref_max_sum':
                print(f'{k:<24s} {v/total_scenes:.9f}')
        print(f'{"max_pred_ref":<24s} {max_pred_ref:.9f}')
        print('sample raw first/mid/last:', first_example['raw_first'], first_example['raw_mid'], first_example['raw_last'])
        print('sample effective residual last:', first_example['effective_last'])
        print('Interpretation: raw_abs large + effective_abs tiny => near-constant residual canceled by recentering.')
        print('If both raw_abs and effective_abs tiny => head itself outputs nearly zero.')
        print('If effective_abs significant but ADE unchanged => output correction is inaccurate, not canceled.')


if __name__ == '__main__':
    main()

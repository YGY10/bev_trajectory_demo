#!/usr/bin/env python3
"""Measure loss-component gradients without training or modifying a checkpoint.

Uses the existing V5 strict exit mask through ExitSupervisedDataset, and the
existing model compatibility loader. No model architecture internals are needed
for the default output-gradient diagnostics. Parameter-gradient diagnostics use
all trainable parameters by default, or the optional named parameter subset.
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train_v5_exit_lane_loss_v2 import ExitSupervisedDataset, exit_lateral_stats
from tools.model_compat import load_trajectory_model


def norms_and_cosines(grads):
    names = list(grads)
    result = {}
    for name, parts in grads.items():
        norm2 = sum(float(torch.sum(x.detach().float().square()).item()) for x in parts if x is not None)
        result[f'grad_norm_{name}'] = math.sqrt(norm2)
    for a, b in [('fit', 'exit'), ('base', 'exit'), ('arc', 'exit'), ('bend', 'exit')]:
        ga, gb = grads[a], grads[b]
        dot = sum(float((x.detach().float() * y.detach().float()).sum().item())
                  for x, y in zip(ga, gb) if x is not None and y is not None)
        na = result[f'grad_norm_{a}']
        nb = result[f'grad_norm_{b}']
        result[f'grad_cos_{a}_vs_{b}'] = dot / (na * nb) if na > 1e-12 and nb > 1e-12 else float('nan')
    return result


def summarize(rows, label):
    print(f'\n[{label}] n={len(rows)}')
    if not rows:
        return
    keys = ['loss_fit', 'loss_arc_weighted', 'loss_bend_weighted', 'loss_exit_weighted',
            'grad_norm_fit', 'grad_norm_arc', 'grad_norm_bend', 'grad_norm_exit',
            'grad_norm_base', 'grad_cos_base_vs_exit', 'grad_cos_fit_vs_exit']
    for k in keys:
        x = np.asarray([r[k] for r in rows], dtype=float)
        x = x[np.isfinite(x)]
        print(f'  {k:<29} mean={x.mean():+.6g} median={np.median(x):+.6g}' if len(x)
              else f'  {k:<29} n/a')
    x = np.array([r['grad_cos_base_vs_exit'] for r in rows], float)
    x = x[np.isfinite(x)]
    if len(x):
        print(f'  conflicting base/exit gradient: {(x < 0).sum()}/{len(x)}')


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', default='data/v5_splits/train.jsonl')
    p.add_argument('--generator-seed', type=int, default=20261021)
    p.add_argument('--max-lines', type=int, default=32)
    p.add_argument('--visibility-gap', type=float, default=2.0)
    p.add_argument('--min-visible-points', type=int, default=5)
    p.add_argument('--samples-per-nav', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lambda-lane', type=float, default=0.02)
    p.add_argument('--parameter-filter', default='head', help='Trainable parameter name substring; default=head (lower VRAM); empty string selects all parameters')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output-dir', default='outputs/loss_gradient_diagnostic')
    a = p.parse_args()
    if a.samples_per_nav < 1 or a.lambda_lane < 0:
        p.error('samples-per-nav must be positive; lambda-lane must be non-negative')

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    ds = ExitSupervisedDataset(a.data, seed=a.generator_seed, max_lines=a.max_lines,
                               visibility_gap=a.visibility_gap, min_visible_points=a.min_visible_points,
                               augment=False, ref_augment=False)
    model, ckpt, info = load_trajectory_model(a.checkpoint, a.device)
    model.eval()
    if int(ckpt.get('horizon', ds.path_points)) != ds.path_points:
        raise ValueError('Checkpoint horizon and GT path length mismatch')
    named = [(n, x) for n, x in model.named_parameters()
             if x.requires_grad and a.parameter_filter in n]
    if not named:
        raise ValueError('No parameters match --parameter-filter')
    parameters = [x for _, x in named]
    print('Model:', info, '| parameter scope:', a.parameter_filter or 'ALL',
          '| tensors:', len(parameters), '| elements:', sum(t.numel() for t in parameters))
    print('Gradient loss components: fit, 0.3*arc, 0.02*bend, lambda_lane*exit.')
    print('All analyses use UN-AUGMENTED dataset items and strict visible exit masks.')

    rng = np.random.default_rng(a.seed)
    bynav = defaultdict(list)
    for i, (row, supervision) in enumerate(zip(ds.rows, ds.exit_supervision)):
        if int(supervision[0].sum().item()) >= a.min_visible_points:
            bynav[row['navigation']].append(i)
    selected = []
    for nav in ('straight', 'left', 'right'):
        choices = bynav[nav]
        if len(choices) < a.samples_per_nav:
            print(f'Warning: {nav} has only {len(choices)} eligible scenes')
        ids = rng.choice(choices, size=min(a.samples_per_nav, len(choices)), replace=False).tolist() if choices else []
        selected += [(nav, i) for i in ids]
    results = []
    for k, (nav, i) in enumerate(selected, 1):
        item = ds[i]
        batch = {name: value.unsqueeze(0).to(a.device) for name, value in item.items()}
        inputs = {name: batch[name] for name in ('lines', 'line_valid', 'reference', 'navigation')}
        with torch.enable_grad():
            pred = model(inputs)
            gt = batch['target_xy']
            fit = F.smooth_l1_loss(pred, gt, beta=1.0)
            dp = pred[:, 1:] - pred[:, :-1]
            arc = 0.3 * (torch.linalg.vector_norm(dp, dim=-1) - ds.spacing).abs().mean()
            bend = 0.02 * (pred[:, 2:] - 2*pred[:, 1:-1] + pred[:, :-2]).square().sum(-1).mean()
            lane, *_ = exit_lateral_stats(pred, batch)
            exit_term = a.lambda_lane * lane
            base = fit + arc + bend
            losses = {'fit': fit, 'arc': arc, 'bend': bend,
                      'exit': exit_term, 'base': base}
            # Output-space gradients: architecture-independent diagnostic.
            output_grads = {name: torch.autograd.grad(value, pred, retain_graph=True)[0].detach()
                            for name, value in losses.items()}
            # Actual parameter-space gradients: measured over selected model parameters.
            parameter_grads = {}
            for name, value in losses.items():
                gg = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
                parameter_grads[name] = [x.detach() if x is not None else None for x in gg]
            out_stats = norms_and_cosines({n: [g] for n, g in output_grads.items()})
            par_stats = norms_and_cosines(parameter_grads)
            row = {'scene_id': ds.rows[i]['id'], 'navigation': nav,
                   'distance_m': float(ds.rows[i]['metadata']['ego_distance_to_junction_m']),
                   'visible_points': int(batch['exit_mask'].sum().item()),
                   'loss_fit': float(fit.detach()), 'loss_arc_weighted': float(arc.detach()),
                   'loss_bend_weighted': float(bend.detach()),
                   'loss_exit_weighted': float(exit_term.detach())}
            row.update({f'output_{key}': value for key, value in out_stats.items()})
            row.update(par_stats)
            results.append(row)
        print(f'{k:02d}/{len(selected)} {nav:8s} {row["scene_id"]:38s} '
              f'param |g_base|={row["grad_norm_base"]:.5g} |g_exit|={row["grad_norm_exit"]:.5g} '
              f'cos={row["grad_cos_base_vs_exit"]:+.3f}', flush=True)
        del batch, pred, losses, output_grads, parameter_grads
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/'per_scene.csv', results)
    for nav in ('straight', 'left', 'right'):
        summarize([r for r in results if r['navigation'] == nav], nav)
    summarize(results, 'ALL')
    print('\nOutput-space gradient checks (independent of network parameters):')
    for nav in ('straight', 'left', 'right'):
        rr = [r for r in results if r['navigation'] == nav]
        if rr:
            print(f'  {nav:<8} mean output cos(base,exit)={np.nanmean([r["output_grad_cos_base_vs_exit"] for r in rr]):+.4f}')
    print('Saved:', out/'per_scene.csv')
    print('NOTE: Small samples and parameter scope can change the results. Negative cosine implies local gradient conflict, not proof of its causal role in training.')

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Black-box conditional sensitivity to smooth Reference perturbations.

Keep BEV and navigation fixed; perturb ONLY reference with an origin-anchored,
smooth lateral shift. Decompose output into the unavoidable reference skip path
and the learned residual response. Also summarize reference-vs-GT errors.

Run from the project root. Requires the existing BEVDataset and checkpoint-aware
loader tools.model_compat.load_trajectory_model. Does not inspect model internals.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import BEVDataset
from tools.model_compat import load_trajectory_model

KEYS = ("lines", "line_valid", "reference", "navigation")
NAVS = ("all", "straight", "left", "right")


def normal_and_profile(reference, profile):
    """Polyline unit normal and dimensionless spatial perturbation profile.

    Both are in the model's ego-relative XY coordinates. First displacement
    is exactly zero; nominal amplitude is at the final reference point.
    """
    tangent = torch.empty_like(reference)
    tangent[1:-1] = reference[2:] - reference[:-2]
    tangent[0] = reference[1] - reference[0]
    tangent[-1] = reference[-1] - reference[-2]
    magnitude = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    if bool((magnitude < 1e-6).any()):
        raise ValueError("Degenerate reference point tangent; cannot define lateral normal")
    t = tangent / magnitude
    normal = torch.stack((-t[:, 1], t[:, 0]), dim=-1)
    n = reference.shape[0]
    s = torch.linspace(0., 1., n, device=reference.device, dtype=reference.dtype)
    if profile == "smoothstep":
        weight = s * s * (3 - 2 * s)
    elif profile == "late":
        z = ((s - 0.50) / 0.50).clamp(0, 1)
        weight = z * z * (3 - 2 * z)
    else:
        raise ValueError(profile)
    weight[0] = 0.
    return normal, weight


def grouped_summary(rows, keys):
    outputs = []
    for nav in NAVS:
        for delta in sorted(set(r['delta_m'] for r in rows)):
            sub = [r for r in rows if r['delta_m'] == delta and (nav == 'all' or r['navigation'] == nav)]
            if not sub:
                continue
            out = {'navigation': nav, 'delta_m': delta, 'n': len(sub)}
            for key in keys:
                vals = np.asarray([r[key] for r in sub], dtype=float)
                out[f'{key}_mean'] = float(np.mean(vals))
                out[f'{key}_median'] = float(np.median(vals))
            outputs.append(out)
    return outputs


def save_csv(path, records):
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', default='data/v5_splits/val.jsonl')
    p.add_argument('--max-lines', type=int, default=None)
    p.add_argument('--deltas', nargs='+', type=float, default=[-0.5, -0.25, 0.0, 0.25, 0.5])
    p.add_argument('--profile', choices=['smoothstep', 'late'], default='smoothstep',
                   help='smoothstep=whole-path drift, late=mainly late-path drift')
    p.add_argument('--navigation', choices=NAVS, default='all')
    p.add_argument('--limit', type=int, default=0, help='Maximum scenes in total; 0=all')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output-dir', default='outputs/reference_condition_sensitivity')
    args = p.parse_args()
    if args.limit < 0:
        p.error('--limit must be >=0')
    deltas = sorted(set(float(v) for v in args.deltas) | {0.})
    if not any(d < 0 for d in deltas) or not any(d > 0 for d in deltas):
        p.error('--deltas must include positive and negative offsets')
    model, ckpt, info = load_trajectory_model(args.checkpoint, args.device)
    max_lines = args.max_lines if args.max_lines is not None else int(info.get('max_lines', 32))
    ds = BEVDataset(args.data, split='all', augment=False,
                    max_lines=max_lines, ref_points=int(ckpt['horizon']))
    out_dir = Path(args.output_dir)
    records, baseline = [], []
    device = torch.device(args.device)
    model.eval()
    with torch.inference_mode():
        for j, row in enumerate(ds.rows):
            nav = row['navigation']
            if args.navigation != 'all' and nav != args.navigation:
                continue
            if args.limit and len(baseline) >= args.limit:
                break
            item = ds[j]
            batch = {k: item[k].unsqueeze(0).to(device) for k in KEYS}
            ref = batch['reference'][0]
            gt = item['target_xy'].to(device)
            normal, profile = normal_and_profile(ref, args.profile)
            original = model(batch)[0]
            # The model returns an origin-anchored path. Compare against likewise anchored ref.
            ref_path = ref - ref[:1]
            original_resid = original - ref_path
            ref_ade = torch.linalg.vector_norm(ref_path - gt, dim=-1).mean().item()
            pred_ade = torch.linalg.vector_norm(original - gt, dim=-1).mean().item()
            br = {'scene_id': row['id'], 'navigation': nav,
                  'reference_ade_m': float(ref_ade), 'prediction_ade_m': float(pred_ade),
                  'reference_fde_m': float(torch.linalg.vector_norm(ref_path[-1] - gt[-1]).item()),
                  'prediction_fde_m': float(torch.linalg.vector_norm(original[-1] - gt[-1]).item()),
                  'prediction_correction_m': float(torch.linalg.vector_norm(original_resid, dim=-1).mean().item())}
            baseline.append(br)
            for delta in deltas:
                # BEV and navigation untouched. Smooth ego-frame lateral reference drift.
                displacement = delta * profile[:, None] * normal
                changed = dict(batch)
                changed['reference'] = batch['reference'].clone()
                changed['reference'][0] += displacement
                changed_ref_path = changed['reference'][0] - changed['reference'][0, :1]
                perturbed = model(changed)[0]
                actual_reference_change = changed_ref_path - ref_path
                total_response = perturbed - original
                # Remove the DIRECT ref addition. This measures how learned correction reacts.
                residual_response = ((perturbed - changed_ref_path) - original_resid)
                signed_total = (total_response * normal).sum(-1)
                signed_direct = (actual_reference_change * normal).sum(-1)
                signed_resid = (residual_response * normal).sum(-1)
                mask = profile >= 0.5  # later part of reference, no GT-dependent filtering
                if not bool(mask.any()):
                    raise RuntimeError('Empty later-path mask')
                def mean_norm(x):
                    return float(torch.linalg.vector_norm(x, dim=-1).mean().item())
                records.append({
                    'scene_id': row['id'], 'navigation': nav, 'delta_m': delta,
                    'reference_ade_m': ref_ade, 'baseline_prediction_ade_m': pred_ade,
                    'perturbed_prediction_ade_m': float(torch.linalg.vector_norm(perturbed - gt, dim=-1).mean().item()),
                    'reference_shift_m': mean_norm(actual_reference_change),
                    'prediction_shift_m': mean_norm(total_response),
                    'residual_response_m': mean_norm(residual_response),
                    'direct_lateral_late_m': float(signed_direct[mask].mean().item()),
                    'pred_lateral_late_m': float(signed_total[mask].mean().item()),
                    'residual_lateral_late_m': float(signed_resid[mask].mean().item()),
                    'residual_response_end_lateral_m': float(signed_resid[-1].item()),
                    'max_input_line_change': float((changed['lines'] - batch['lines']).abs().max().item()),
                    'max_input_nav_change': int((changed['navigation'] != batch['navigation']).sum().item()),
                    'output_finite': bool(torch.isfinite(perturbed).all().item()),
                })
    if not baseline:
        raise RuntimeError('No scenes selected')
    if not all(r['output_finite'] for r in records):
        raise RuntimeError('Non-finite model prediction encountered')
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = ['reference_shift_m', 'prediction_shift_m', 'residual_response_m',
            'direct_lateral_late_m', 'pred_lateral_late_m', 'residual_lateral_late_m',
            'perturbed_prediction_ade_m']
    summary = grouped_summary(records, keys)
    save_csv(out_dir / 'per_scene_delta.csv', records)
    save_csv(out_dir / 'summary.csv', summary)
    save_csv(out_dir / 'reference_baseline_per_scene.csv', baseline)
    b = np.array([r['reference_ade_m'] for r in baseline])
    pb = np.array([r['prediction_ade_m'] for r in baseline])
    print(f"Loaded: {args.checkpoint} variant={info.get('model_variant')} "
          f"encoder={info.get('bev_encoder')} token_mode={info.get('bev_token_mode', ckpt.get('bev_token_mode'))}")
    print(f'Scenes={len(baseline)}; profile={args.profile}; reference ADE: mean={b.mean():.6f} '
          f'median={np.median(b):.6f} p10={np.percentile(b,10):.6f} '
          f'p90={np.percentile(b,90):.6f}; model ADE={pb.mean():.6f}')
    print('Delta | direct ref late / output late / residual late (signed meters) | residual whole-path response (m)')
    for r in summary:
        if r['navigation'] == 'all' and r['delta_m'] != 0.0:
            print(f"{r['delta_m']:+.2f} | {r['direct_lateral_late_m_mean']:+.6f} / "
                  f"{r['pred_lateral_late_m_mean']:+.6f} / "
                  f"{r['residual_lateral_late_m_mean']:+.6f} | "
                  f"{r['residual_response_m_mean']:.6f}")
    print('Saved:', out_dir / 'summary.csv', out_dir / 'reference_baseline_per_scene.csv')
    print('IMPORTANT: reference effect includes a direct skip path. Residual response is the network-induced part.')
    print('Reference ADE statistics describe the EXISTING data, not new or synthetic error settings.')


if __name__ == '__main__':
    main()

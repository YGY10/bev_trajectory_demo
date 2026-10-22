#!/usr/bin/env python3
"""Architecture-agnostic geometry perturbation diagnostic for BEV path predictors.

Contract: model accepts lines/line_valid/reference/navigation; returns [B,H,2].
No hooks, module names, attention internals, or fixed token counts.
Ground-truth exit geometry is used ONLY for selecting perturbations and scoring.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from datasets import BEVDataset
from evaluate_exit_lane_v5 import base_geometry, observed_sides, near_visible
from tools.model_compat import load_trajectory_model

INPUT_KEYS = ("lines", "line_valid", "reference", "navigation")


def target_line_indices(row, anchor, normal, width, max_lines, ratio=0.33):
    elems = row["lane_markings"] + row["road_boundaries"]
    selected = []
    for i, element in enumerate(elems[:max_lines]):
        if element.get("source_road") != "exit":
            continue
        xy = np.asarray(element.get("points", []), dtype=float)
        if xy.ndim != 2 or xy.shape[0] < 2 or xy.shape[1] != 2:
            continue
        lateral = float(np.median((xy - anchor) @ normal))
        if min(abs(lateral - width / 2), abs(lateral + width / 2)) <= width * ratio:
            selected.append(i)
    return selected


def predict(model, tensors):
    output = model(tensors)
    if isinstance(output, dict):
        if "path" not in output:
            raise ValueError("A dict-returning model must provide output['path']")
        output = output["path"]
    if not isinstance(output, torch.Tensor) or output.ndim != 3 or output.shape[-1] != 2:
        raise ValueError(f"Expected path tensor [B,H,2], got {type(output)} and {getattr(output, 'shape', None)}")
    if not torch.isfinite(output).all().item():
        raise ValueError("Model output contains NaN or infinity")
    return output[0].detach().cpu().numpy().astype(np.float64)


def summarize(rows, key):
    v = np.asarray([row[key] for row in rows], dtype=float)
    return f"mean={v.mean():+.9f} median={np.median(v):+.9f} max_abs={np.abs(v).max():.9f}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="data/v5_splits/val.jsonl")
    p.add_argument("--generator-seed", type=int, default=20261021)
    p.add_argument("--max-lines", type=int, default=None,
                   help="Default: max_lines stored in checkpoint")
    p.add_argument("--station-gap-m", type=float, default=1.5)
    p.add_argument("--min-points", type=int, default=5)
    p.add_argument("--target-offset-ratio", type=float, default=0.33)
    p.add_argument("--navigation", choices=("straight", "left", "right", "all"), default="straight")
    p.add_argument("--delta-m", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=30, help="0 for all eligible scenes")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default="outputs/geometry_blackbox")
    args = p.parse_args()
    if not np.isfinite(args.delta_m) or args.delta_m == 0:
        p.error("--delta-m must be finite and nonzero")
    if args.min_points < 1 or args.station_gap_m <= 0 or not 0 < args.target_offset_ratio < 1:
        p.error("Invalid eligibility parameters")
    if args.max_lines is not None and args.max_lines < 1:
        p.error("--max-lines must be positive")

    model, ckpt, info = load_trajectory_model(args.checkpoint, args.device)
    max_lines = args.max_lines if args.max_lines is not None else int(info.get("max_lines", 32))
    dataset = BEVDataset(args.data, split="all", augment=False,
                         max_lines=max_lines, ref_points=int(info["horizon"]))
    print(f"Loaded checkpoint={args.checkpoint} variant={info.get('model_variant')} "
          f"encoder={info.get('bev_encoder')} token_mode={info.get('bev_token_mode', 'unknown')}")

    rows_out = []
    eligible_seen = 0
    with torch.inference_mode():
        for idx, row in enumerate(dataset.rows):
            if args.navigation != "all" and row["navigation"] != args.navigation:
                continue
            item = dataset[idx]
            anchor, tangent, normal, width, lane_index = base_geometry(row, args.generator_seed)
            sides = observed_sides(row, anchor, tangent, normal, width, lane_index)
            gt = item["target_xy"].cpu().numpy().astype(np.float64)
            station = (gt - anchor) @ tangent
            mask = ((station >= 3.0)
                    & near_visible(station, sides[0], args.station_gap_m)
                    & near_visible(station, sides[1], args.station_gap_m))
            if int(mask.sum()) < args.min_points:
                continue
            eligible_seen += 1
            chosen = target_line_indices(row, anchor, normal, width, max_lines,
                                         ratio=args.target_offset_ratio)
            if not chosen:
                continue
            if max(chosen) >= item["lines"].shape[0]:
                raise RuntimeError("Selected raw-line index exceeds BEVDataset line count; check data ordering")
            inputs = {k: item[k].unsqueeze(0).to(args.device) for k in INPUT_KEYS}
            if not inputs["line_valid"][0, chosen].all().item():
                raise RuntimeError("Target line is marked invalid; data/selection ordering mismatch")
            disturbed = {k: v.clone() for k, v in inputs.items()}
            shift = torch.tensor(normal * args.delta_m,
                                 device=args.device, dtype=disturbed["lines"].dtype)
            # Modify ONLY x/y of observed line points. Tangent feature stays unchanged.
            # Indexing first the 0-th batch prevents accidental advanced-index reshaping.
            disturbed["lines"][0, chosen, :, :2] += shift
            displacement = disturbed["lines"][0, chosen, :, :2] - inputs["lines"][0, chosen, :, :2]
            err = (displacement - shift).abs().max().item()
            if err > 1e-4:
                raise RuntimeError(f"Input shift precision error {err}")
            if not torch.equal(inputs["reference"], disturbed["reference"]):
                raise RuntimeError("Reference changed during BEV perturbation")
            if not torch.equal(inputs["navigation"], disturbed["navigation"]):
                raise RuntimeError("Navigation changed during BEV perturbation")
            a = predict(model, inputs)
            b = predict(model, disturbed)
            if a.shape != gt.shape or b.shape != gt.shape:
                raise ValueError(f"GT/pred mismatch: gt={gt.shape}, pred={a.shape}")
            movement = b - a
            signed = movement[mask] @ normal
            rows_out.append({
                "scene_id": row["id"], "navigation": row["navigation"],
                "selected_lines": len(chosen), "eligible_exit_points": int(mask.sum()),
                "delta_m": args.delta_m,
                "input_translation_max_error_m": float(err),
                "input_actual_shift_mean_m": float(torch.linalg.vector_norm(displacement, dim=-1).mean().item()),
                "exit_signed_response_m": float(signed.mean()),
                "exit_absolute_response_m": float(np.abs(signed).mean()),
                "exit_path_shift_m": float(np.linalg.norm(movement[mask], axis=-1).mean()),
                "whole_path_shift_m": float(np.linalg.norm(movement, axis=-1).mean()),
                "exit_slope": float(signed.mean() / args.delta_m),
                "baseline_ade_m": float(np.linalg.norm(a - gt, axis=-1).mean()),
                "perturbed_ade_m": float(np.linalg.norm(b - gt, axis=-1).mean()),
            })
            if args.limit > 0 and len(rows_out) >= args.limit:
                break

    if not rows_out:
        raise RuntimeError("No eligible scenes with matching target exit polylines")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_scene.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Checked={len(rows_out)}, eligible_seen={eligible_seen}, delta={args.delta_m:+.3f}m")
    for metric in ("input_actual_shift_mean_m", "input_translation_max_error_m",
                   "exit_signed_response_m", "exit_absolute_response_m",
                   "exit_path_shift_m", "whole_path_shift_m", "exit_slope",
                   "baseline_ade_m", "perturbed_ade_m"):
        print(f"{metric:32s} {summarize(rows_out, metric)}")
    print(f"Saved: {out / 'per_scene.csv'}")
    print("Note: sensitivity is not proof of using the BEV geometry correctly.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Targeted exit-road-line ablation for a trained BEV path model."""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from datasets.bev_dataset import BEVDataset
from tools.model_compat import load_trajectory_model


def avg(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-lines", type=int, default=32)
    parser.add_argument("--navigation", choices=["all", "straight", "left", "right"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="0 means all matching scenes")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, info = load_trajectory_model(args.checkpoint, device)
    model.eval()
    print(f"Loaded model={info.get('model_variant', 'unknown')} device={device}")

    dataset = BEVDataset(
        args.data, split="all", seed=args.seed,
        max_lines=args.max_lines, augment=False,
    )
    indices = [
        i for i, row in enumerate(dataset.rows)
        if args.navigation == "all" or row["navigation"] == args.navigation
    ]
    if args.limit > 0:
        indices = indices[:args.limit]
    if not indices:
        raise RuntimeError("No matching scenes.")

    conditions = [
        "remove_exit_all",
        "remove_exit_lane_markings",
        "remove_exit_road_boundaries",
        "matched_drop_entry",
    ]
    records = []
    skipped_no_exit = 0
    skipped_no_entry = 0

    for count, index in enumerate(indices, 1):
        row = dataset.rows[index]
        item = dataset[index]
        batch = {
            k: v.unsqueeze(0).to(device)
            for k, v in item.items() if torch.is_tensor(v)
        }

        # Mirrors BEVDataset.__getitem__ with augment=False:
        # lane_markings followed by road_boundaries, then max_lines truncation.
        elements = (
            [(e, "lane_marking") for e in row.get("lane_markings", [])]
            + [(e, "road_boundary") for e in row.get("road_boundaries", [])]
        )[:args.max_lines]

        valid_ids = set(torch.where(batch["line_valid"][0].bool())[0].tolist())
        exit_ids = [
            i for i, (e, _) in enumerate(elements)
            if e.get("source_road") == "exit" and i in valid_ids
        ]
        entry_ids = [
            i for i, (e, _) in enumerate(elements)
            if e.get("source_road") == "entry" and i in valid_ids
        ]
        exit_lane_ids = [
            i for i, (e, kind) in enumerate(elements)
            if e.get("source_road") == "exit" and kind == "lane_marking" and i in valid_ids
        ]
        exit_boundary_ids = [
            i for i, (e, kind) in enumerate(elements)
            if e.get("source_road") == "exit" and kind == "road_boundary" and i in valid_ids
        ]

        if not exit_ids:
            skipped_no_exit += 1
            continue
        if not entry_ids:
            skipped_no_entry += 1

        # Count-matched control: remove the same number of entry lines as exit
        # lines, spread deterministically across the available entry lines.
        if entry_ids:
            k = min(len(exit_ids), len(entry_ids))
            positions = np.linspace(0, len(entry_ids) - 1, num=k, dtype=int)
            matched_entry_ids = [entry_ids[p] for p in positions]
        else:
            matched_entry_ids = []

        with torch.inference_mode():
            baseline = model(batch)
            if not torch.isfinite(baseline).all():
                raise RuntimeError(f"Non-finite baseline prediction: {row['id']}")
            gt = batch["target_xy"]
            baseline_ade = torch.linalg.vector_norm(baseline - gt, dim=-1).mean().item()

            drop_sets = {
                "remove_exit_all": exit_ids,
                "remove_exit_lane_markings": exit_lane_ids,
                "remove_exit_road_boundaries": exit_boundary_ids,
                "matched_drop_entry": matched_entry_ids,
            }
            for condition in conditions:
                dropped = drop_sets[condition]
                if not dropped:
                    continue

                test = {k: v.clone() for k, v in batch.items()}
                test["line_valid"][0, dropped] = False
                test["lines"][0, dropped, :, :] = 0
                pred = model(test)
                if not torch.isfinite(pred).all():
                    raise RuntimeError(f"Non-finite prediction: {row['id']} / {condition}")

                shift = torch.linalg.vector_norm(pred - baseline, dim=-1)[0]
                condition_ade = torch.linalg.vector_norm(pred - gt, dim=-1).mean().item()
                records.append({
                    "scene_id": row["id"],
                    "navigation": row["navigation"],
                    "condition": condition,
                    "baseline_ade_m": baseline_ade,
                    "condition_ade_m": condition_ade,
                    "delta_ade_m": condition_ade - baseline_ade,
                    "whole_path_shift_m": shift.mean().item(),
                    "early_path_shift_m": shift[:26].mean().item(),
                    "late_path_shift_m": shift[50:].mean().item(),
                    "final_point_shift_m": shift[-1].item(),
                    "retained_exit_lines": len(exit_ids),
                    "retained_exit_lane_markings": len(exit_lane_ids),
                    "retained_exit_road_boundaries": len(exit_boundary_ids),
                    "retained_entry_lines": len(entry_ids),
                    "lines_removed": len(dropped),
                    "valid_lines_before": len(valid_ids),
                    "valid_lines_after": int(test["line_valid"].sum().item()),
                })

        if count % 10 == 0 or count == len(indices):
            print(f"Processed {count}/{len(indices)} scenes")

    if not records:
        raise RuntimeError(
            "No scenes with retained exit lines were evaluated; check source_road and --max-lines."
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fields = list(records[0].keys())
    with (out / "per_scene.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    summary_fields = [
        "navigation", "condition", "n", "mean_baseline_ade_m",
        "mean_condition_ade_m", "mean_delta_ade_m",
        "mean_whole_path_shift_m", "mean_early_path_shift_m",
        "mean_late_path_shift_m", "mean_final_point_shift_m",
        "mean_lines_removed", "mean_retained_exit_lines",
    ]
    summary = []
    for nav in ["all", "straight", "left", "right"]:
        nav_rows = records if nav == "all" else [r for r in records if r["navigation"] == nav]
        for condition in conditions:
            group = [r for r in nav_rows if r["condition"] == condition]
            if not group:
                continue
            summary.append({
                "navigation": nav,
                "condition": condition,
                "n": len(group),
                "mean_baseline_ade_m": avg(group, "baseline_ade_m"),
                "mean_condition_ade_m": avg(group, "condition_ade_m"),
                "mean_delta_ade_m": avg(group, "delta_ade_m"),
                "mean_whole_path_shift_m": avg(group, "whole_path_shift_m"),
                "mean_early_path_shift_m": avg(group, "early_path_shift_m"),
                "mean_late_path_shift_m": avg(group, "late_path_shift_m"),
                "mean_final_point_shift_m": avg(group, "final_point_shift_m"),
                "mean_lines_removed": avg(group, "lines_removed"),
                "mean_retained_exit_lines": avg(group, "retained_exit_lines"),
            })

    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary)

    print("\n===== Exit-line ablation summary =====")
    print("navigation condition                       n  ADE_before ADE_after dADE      whole      early      late       final")
    for r in summary:
        print(
            f"{r['navigation']:10s} {r['condition']:31s} {r['n']:3d} "
            f"{r['mean_baseline_ade_m']:.4f} {r['mean_condition_ade_m']:.4f} "
            f"{r['mean_delta_ade_m']:+.4f} {r['mean_whole_path_shift_m']:.5f} "
            f"{r['mean_early_path_shift_m']:.5f} {r['mean_late_path_shift_m']:.5f} "
            f"{r['mean_final_point_shift_m']:.5f}"
        )
    print(f"\nScenes requested: {len(indices)}")
    print(f"Scenes evaluated: {len(set(r['scene_id'] for r in records))}")
    print(f"Skipped without retained exit lines: {skipped_no_exit}")
    print(f"Scenes without entry-line control: {skipped_no_entry}")
    print(f"Saved: {out / 'per_scene.csv'}")
    print(f"Saved: {out / 'summary.csv'}")
    print("No model weights were changed.")


if __name__ == "__main__":
    main()

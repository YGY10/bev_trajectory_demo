#!/usr/bin/env python3
"""Black-box BEV feature ablation; does not inspect model internals."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from datasets.bev_dataset import BEVDataset
from tools.model_compat import load_trajectory_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-lines", type=int, default=32)
    parser.add_argument("--navigation",
                        choices=["all", "straight", "left", "right"],
                        default="all")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, ckpt, info = load_trajectory_model(
        args.checkpoint, device
    )
    model.eval()

    print(
        f"Loaded model={info.get('model_variant', 'unknown')} "
        f"device={device}"
    )

    dataset = BEVDataset(
        args.data,
        split="all",
        seed=args.seed,
        max_lines=args.max_lines,
        augment=False,
    )

    indices = [
        i for i, row in enumerate(dataset.rows)
        if args.navigation == "all"
        or row["navigation"] == args.navigation
    ]
    if args.limit:
        indices = indices[:args.limit]

    if not indices:
        raise RuntimeError("No matching scenes.")

    names = [
        "baseline",
        "xy_zero",
        "geometry_zero",
        "semantics_zero",
        "all_features_zero",
        "drop_half_lines",
    ]

    records = []

    for count, index in enumerate(indices, 1):
        row = dataset.rows[index]
        item = dataset[index]

        batch = {
            k: v.unsqueeze(0).to(device)
            for k, v in item.items()
            if torch.is_tensor(v)
        }

        valid = batch["line_valid"][0].bool()
        valid_ids = torch.where(valid)[0]

        if len(valid_ids) == 0:
            continue

        with torch.inference_mode():
            baseline = model(batch)

            if not torch.isfinite(baseline).all():
                raise RuntimeError(
                    f"Non-finite baseline: {row['id']}"
                )

            gt = batch["target_xy"]
            baseline_ade = torch.linalg.vector_norm(
                baseline - gt, dim=-1
            ).mean().item()

            for name in names[1:]:
                test = {
                    k: v.clone() for k, v in batch.items()
                }
                lines = test["lines"]

                if name == "xy_zero":
                    lines[0, valid, :, :2] = 0

                elif name == "geometry_zero":
                    lines[0, valid, :, :4] = 0

                elif name == "semantics_zero":
                    lines[0, valid, :, 4:11] = 0

                elif name == "all_features_zero":
                    lines[0, valid, :, :] = 0

                elif name == "drop_half_lines":
                    # Keep every other valid line.
                    dropped = valid_ids[1::2]
                    test["line_valid"][0, dropped] = False
                    lines[0, dropped, :, :] = 0

                pred = model(test)

                if not torch.isfinite(pred).all():
                    raise RuntimeError(
                        f"Non-finite prediction: "
                        f"{row['id']} / {name}"
                    )

                displacement = torch.linalg.vector_norm(
                    pred - baseline, dim=-1
                )[0]

                ade = torch.linalg.vector_norm(
                    pred - gt, dim=-1
                ).mean().item()

                records.append({
                    "scene_id": row["id"],
                    "navigation": row["navigation"],
                    "condition": name,
                    "baseline_ade_m": baseline_ade,
                    "condition_ade_m": ade,
                    "delta_ade_m": ade - baseline_ade,
                    "whole_path_shift_m": displacement.mean().item(),
                    "late_path_shift_m": displacement[50:].mean().item(),
                    "final_point_shift_m": displacement[-1].item(),
                    "valid_lines_before": len(valid_ids),
                    "valid_lines_after": int(
                        test["line_valid"].sum().item()
                    ),
                })

        if count % 10 == 0 or count == len(indices):
            print(f"Processed {count}/{len(indices)}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fields = list(records[0].keys())
    with (out / "per_scene.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    print("\n===== BEV ablation summary =====")
    print(
        "condition              n   ADE_before ADE_after "
        "dADE      path_shift  late_shift  final_shift"
    )

    for name in names[1:]:
        group = [r for r in records if r["condition"] == name]
        if not group:
            continue

        mean = lambda key: np.mean([r[key] for r in group])

        print(
            f"{name:22s} {len(group):3d} "
            f"{mean('baseline_ade_m'):.4f} "
            f"{mean('condition_ade_m'):.4f} "
            f"{mean('delta_ade_m'):+.4f} "
            f"{mean('whole_path_shift_m'):.6f} "
            f"{mean('late_path_shift_m'):.6f} "
            f"{mean('final_point_shift_m'):.6f}"
        )

    print(f"\nSaved: {out / 'per_scene.csv'}")
    print("No model weights were changed.")


if __name__ == "__main__":
    main()

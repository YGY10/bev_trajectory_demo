#!/usr/bin/env python3
"""Architecture-agnostic end-to-end BEV input sensitivity diagnostic.

Keep the reference, navigation, GT, and model weights fixed.
Shift only valid BEV polyline coordinates laterally.
Measure the resulting changes in the predicted path.
"""

import argparse
import csv
from collections import defaultdict
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
    parser.add_argument("--deltas", type=float,
                        nargs="+",
                        default=[-1.0, -0.5, 0.5, 1.0])
    parser.add_argument("--late-start", type=int, default=50)
    args = parser.parse_args()

    if args.max_lines <= 0 or args.limit < 0:
        parser.error("max-lines must be positive; limit must be >= 0")
    if not args.deltas or any(abs(d) < 1e-9 for d in args.deltas):
        parser.error("deltas must contain nonzero shifts")
    if not 1 <= args.late_start < 81:
        parser.error("late-start must be in [1, 80]")

    device = args.device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, ckpt, info = load_trajectory_model(
        args.checkpoint, device
    )
    model.eval()

    print(
        f"Loaded checkpoint={args.checkpoint} "
        f"variant={info.get('model_variant', 'unknown')} "
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
        raise RuntimeError("No scenes match the requested filters.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    late_start = args.late_start

    for count, index in enumerate(indices, 1):
        row = dataset.rows[index]
        item = dataset[index]

        batch = {
            key: value.unsqueeze(0).to(device)
            for key, value in item.items()
            if torch.is_tensor(value)
        }

        lines = batch["lines"]
        valid = batch["line_valid"].bool()

        # Select real polyline points only; never shift padding.
        point_mask = valid.unsqueeze(-1).expand(
            -1, -1, lines.shape[2]
        )

        with torch.inference_mode():
            baseline = model(batch)

            if not torch.isfinite(baseline).all():
                raise RuntimeError(
                    f"Non-finite baseline prediction: {row['id']}"
                )

            gt = batch["target_xy"]
            base_ade = torch.linalg.vector_norm(
                baseline - gt, dim=-1
            ).mean().item()

            for delta in args.deltas:
                perturbed = {
                    key: value.clone()
                    for key, value in batch.items()
                }

                # Ego coordinates: x forward, y left.
                # Shift all valid BEV points along y.
                # Keep tangent/direction channels unchanged because
                # rigid translation does not change line direction.
                y = perturbed["lines"][..., 1]
                before = y[point_mask].clone()
                y[point_mask] += delta
                after = y[point_mask]

                actual_shift = (
                    after - before
                ).abs().mean().item()

                if not np.isclose(
                    actual_shift, abs(delta), atol=1e-5
                ):
                    raise RuntimeError(
                        f"Incorrect input shift in scene {row['id']}"
                    )

                pred = model(perturbed)

                if not torch.isfinite(pred).all():
                    raise RuntimeError(
                        f"Non-finite perturbed prediction: "
                        f"{row['id']}, delta={delta}"
                    )

                # Difference between two predicted paths.
                dp = pred - baseline
                point_shift = torch.linalg.vector_norm(
                    dp, dim=-1
                )[0]

                path_shift = point_shift.mean().item()
                late_shift = point_shift[late_start:].mean().item()
                final_shift = point_shift[-1].item()

                lateral = dp[0, :, 1]
                late_lateral = lateral[late_start:]

                pert_ade = torch.linalg.vector_norm(
                    pred - gt, dim=-1
                ).mean().item()

                records.append({
                    "scene_id": row["id"],
                    "navigation": row["navigation"],
                    "distance_m": float(
                        row.get("metadata", {}).get(
                            "ego_distance_to_junction_m",
                            float("nan"),
                        )
                    ),
                    "delta_m": float(delta),
                    "actual_input_shift_m": actual_shift,
                    "baseline_ade_m": base_ade,
                    "perturbed_ade_m": pert_ade,
                    "delta_ade_m": pert_ade - base_ade,
                    "whole_path_shift_m": path_shift,
                    "late_path_shift_m": late_shift,
                    "final_point_shift_m": final_shift,
                    "mean_signed_lateral_response_m": (
                        lateral.mean().item()
                    ),
                    "late_signed_lateral_response_m": (
                        late_lateral.mean().item()
                    ),
                    "mean_abs_lateral_response_m": (
                        lateral.abs().mean().item()
                    ),
                    "lateral_gain": (
                        lateral.abs().mean().item() / abs(delta)
                    ),
                })

        if count % 10 == 0 or count == len(indices):
            print(f"Processed {count}/{len(indices)} scenes")

    fields = list(records[0].keys())
    per_scene_path = output_dir / "per_scene.csv"

    with per_scene_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    # Summarize by navigation and intervention magnitude.
    groups = defaultdict(list)
    for record in records:
        groups[("all", record["delta_m"])].append(record)
        groups[(record["navigation"], record["delta_m"])].append(
            record
        )

    summary_fields = [
        "navigation", "delta_m", "n",
        "baseline_ade_m", "perturbed_ade_m", "delta_ade_m",
        "whole_path_shift_m", "late_path_shift_m",
        "final_point_shift_m",
        "mean_abs_lateral_response_m", "lateral_gain",
    ]
    summaries = []

    for (nav, delta), group in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        result = {
            "navigation": nav,
            "delta_m": delta,
            "n": len(group),
        }
        for key in summary_fields[3:]:
            result[key] = float(np.mean([
                r[key] for r in group
            ]))
        summaries.append(result)

    summary_path = output_dir / "summary.csv"
    with summary_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=summary_fields
        )
        writer.writeheader()
        writer.writerows(summaries)

    print("\n===== End-to-end BEV input sensitivity =====")
    print(
        "navigation delta_m n "
        "ADE_before ADE_after dADE "
        "whole_shift late_shift final_shift lateral_gain"
    )

    for r in summaries:
        if r["navigation"] != "all":
            continue
        print(
            f'{r["navigation"]:>5} '
            f'{r["delta_m"]:+6.2f} '
            f'{r["n"]:3d} '
            f'{r["baseline_ade_m"]:.4f} '
            f'{r["perturbed_ade_m"]:.4f} '
            f'{r["delta_ade_m"]:+.4f} '
            f'{r["whole_path_shift_m"]:.6f} '
            f'{r["late_path_shift_m"]:.6f} '
            f'{r["final_point_shift_m"]:.6f} '
            f'{r["lateral_gain"]:.6f}'
        )

    print(f"\nSaved: {per_scene_path}")
    print(f"Saved: {summary_path}")
    print("No model weights were changed.")


if __name__ == "__main__":
    main()

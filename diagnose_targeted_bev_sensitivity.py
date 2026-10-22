"""Targeted entry/exit BEV geometry intervention.

Keeps checkpoint, reference, navigation and GT fixed.
Perturbs only entry or exit BEV polylines along their local normals.
Does not train or modify model weights.
"""

import argparse
import copy
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from datasets.bev_dataset import BEVDataset
from tools.model_compat import load_trajectory_model


def local_normal_shift(points, delta_m):
    """Shift a polyline along its local left-hand normal."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2).copy()

    if len(p) < 2:
        raise ValueError("A polyline must contain at least two points")

    tangent = np.gradient(p, axis=0)
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = tangent / np.maximum(norm, 1e-8)

    normal = np.stack(
        [-tangent[:, 1], tangent[:, 0]], axis=-1
    )

    return (p + float(delta_m) * normal).tolist()


def perturb_row(row, target_road, delta_m, max_lines):
    """Modify only selected road lines that enter the model tensor."""
    changed = copy.deepcopy(row)

    # Match BEVDataset's actual line ordering and truncation.
    elems = (
        changed["lane_markings"]
        + changed["road_boundaries"]
    )[:max_lines]

    selected = [
        e for e in elems
        if e.get("source_road") == target_road
    ]

    if not selected:
        return changed, 0

    for elem in selected:
        elem["points"] = local_normal_shift(
            elem["points"], delta_m
        )

    return changed, len(selected)


def make_batch(sample, device):
    return {
        key: value.unsqueeze(0).to(device)
        for key, value in sample.items()
    }


def predict(model, batch):
    with torch.inference_mode():
        pred = model(batch)

    if pred.ndim != 3 or pred.shape[0] != 1:
        raise RuntimeError(
            f"Unexpected prediction shape: {tuple(pred.shape)}"
        )

    return pred[0].detach().cpu().numpy()


def path_metrics(pred, gt):
    error = np.linalg.norm(pred - gt, axis=-1)
    return float(error.mean()), float(error[-1])


def response_metrics(base, perturbed, gt, target_road, delta):
    displacement = perturbed - base
    magnitude = np.linalg.norm(displacement, axis=-1)

    # Use GT path tangent to define a consistent lateral direction.
    tangent = np.gradient(gt, axis=0)
    tangent /= np.maximum(
        np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-8
    )
    normal = np.stack(
        [-tangent[:, 1], tangent[:, 0]], axis=-1
    )

    signed_lateral = np.sum(displacement * normal, axis=-1)

    # Entry intervention: inspect early path.
    # Exit intervention: inspect late path.
    if target_road == "entry":
        relevant = slice(1, 41)
    else:
        relevant = slice(40, None)

    signed_response = float(signed_lateral[relevant].mean())

    return {
        "whole_shift_m": float(magnitude.mean()),
        "early_shift_m": float(magnitude[:41].mean()),
        "late_shift_m": float(magnitude[40:].mean()),
        "final_shift_m": float(magnitude[-1]),
        "signed_lateral_response_m": signed_response,
        "lateral_gain": (
            signed_response / delta if abs(delta) > 1e-9
            else float("nan")
        ),
    }


def write_csv(path, rows):
    if not rows:
        return

    with Path(path).open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--max-lines", type=int, default=32)
    parser.add_argument(
        "--navigation",
        choices=["all", "straight", "left", "right"],
        default="all",
    )
    parser.add_argument(
        "--target-road",
        choices=["entry", "exit", "both"],
        default="both",
    )
    parser.add_argument(
        "--deltas", nargs="+", type=float,
        default=[-1.0, -0.5, 0.5, 1.0],
    )
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output-dir",
        default="outputs/targeted_bev_sensitivity",
    )

    args = parser.parse_args()

    if args.max_lines < 1 or args.limit < 1:
        parser.error("--max-lines and --limit must be positive")

    device = args.device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, ckpt, info = load_trajectory_model(
        args.checkpoint, device
    )
    model.to(device)
    model.eval()

    print(
        f"Loaded checkpoint={args.checkpoint} "
        f"device={device}"
    )

    # split=all retains every scene. No augmentation is applied.
    dataset = BEVDataset(
        args.data,
        split="all",
        augment=False,
        max_lines=args.max_lines,
    )

    indices = [
        i for i, row in enumerate(dataset.rows)
        if args.navigation == "all"
        or row.get("navigation") == args.navigation
    ][:args.limit]

    if not indices:
        raise RuntimeError("No scenes matched the selection")

    targets = (
        ["entry", "exit"]
        if args.target_road == "both"
        else [args.target_road]
    )

    records = []
    skipped = defaultdict(int)

    for count, idx in enumerate(indices, start=1):
        original_row = dataset.rows[idx]
        scene_id = original_row.get("id", str(idx))
        navigation = original_row.get("navigation", "unknown")

        # Establish baseline once per scene.
        base_sample = dataset[idx]
        base_batch = make_batch(base_sample, device)
        base_pred = predict(model, base_batch)
        gt = base_sample["target_xy"].numpy()

        ade_before, fde_before = path_metrics(base_pred, gt)

        for target_road in targets:
            for delta in args.deltas:
                if abs(delta) < 1e-9:
                    continue

                changed_row, n_lines = perturb_row(
                    original_row,
                    target_road,
                    delta,
                    args.max_lines,
                )

                if n_lines == 0:
                    skipped[target_road] += 1
                    continue

                try:
                    dataset.rows[idx] = changed_row
                    changed_sample = dataset[idx]
                finally:
                    dataset.rows[idx] = original_row

                changed_batch = make_batch(
                    changed_sample, device
                )
                perturbed_pred = predict(model, changed_batch)

                ade_after, fde_after = path_metrics(
                    perturbed_pred, gt
                )
                response = response_metrics(
                    base_pred, perturbed_pred, gt,
                    target_road, delta,
                )

                records.append({
                    "scene_id": scene_id,
                    "navigation": navigation,
                    "target_road": target_road,
                    "delta_m": delta,
                    "n_target_lines": n_lines,
                    "ade_before_m": ade_before,
                    "ade_after_m": ade_after,
                    "dADE_m": ade_after - ade_before,
                    "fde_before_m": fde_before,
                    "fde_after_m": fde_after,
                    **response,
                })

        if count % 10 == 0 or count == len(indices):
            print(f"Processed {count}/{len(indices)} scenes")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / "per_scene.csv", records)

    # Macro means over scenes, grouped by road, navigation and delta.
    groups = defaultdict(list)
    for row in records:
        key = (
            row["target_road"],
            row["navigation"],
            row["delta_m"],
        )
        groups[key].append(row)

    summary = []
    metric_names = [
        "ade_before_m",
        "ade_after_m",
        "dADE_m",
        "fde_before_m",
        "fde_after_m",
        "whole_shift_m",
        "early_shift_m",
        "late_shift_m",
        "final_shift_m",
        "signed_lateral_response_m",
        "lateral_gain",
    ]

    for (road, nav, delta), rows in sorted(groups.items()):
        item = {
            "target_road": road,
            "navigation": nav,
            "delta_m": delta,
            "n": len(rows),
        }
        for metric in metric_names:
            item[metric] = float(np.mean([
                r[metric] for r in rows
            ]))
        summary.append(item)

    write_csv(out_dir / "summary.csv", summary)

    print("\n===== Targeted BEV sensitivity =====")
    print(
        "road       nav       delta    n    "
        "dADE       whole      early      late       final      "
        "signed_lat   gain"
    )

    for row in summary:
        print(
            f"{row['target_road']:10s} "
            f"{row['navigation']:9s} "
            f"{row['delta_m']:+5.2f} "
            f"{row['n']:4d} "
            f"{row['dADE_m']:+.4f} "
            f"{row['whole_shift_m']:.4f} "
            f"{row['early_shift_m']:.4f} "
            f"{row['late_shift_m']:.4f} "
            f"{row['final_shift_m']:.4f} "
            f"{row['signed_lateral_response_m']:+.4f} "
            f"{row['lateral_gain']:+.4f}"
        )

    print(f"\nSaved: {out_dir}/per_scene.csv")
    print(f"Saved: {out_dir}/summary.csv")

    for road, n in sorted(skipped.items()):
        print(
            f"Skipped {n} scene-interventions for {road}: "
            "no matching road lines among the model's retained lines"
        )

    print(
        "\nDiagnostic only. Positive lateral_gain indicates "
        "response along the chosen local-normal sign; verify "
        "the sign convention and inspect individual scenes."
    )


if __name__ == "__main__":
    main()

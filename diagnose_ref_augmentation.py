import argparse
import csv
import json
import random
from itertools import combinations
from pathlib import Path

import numpy as np

from datasets.bev_dataset import resample, augment_reference_geometry


def load_rows(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(s) for s in f if s.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="data/v5_splits/train.jsonl"
    )
    parser.add_argument("--scene-id", default=None)
    parser.add_argument(
        "--navigation", choices=["straight", "left", "right"],
        default="left"
    )
    parser.add_argument("--distance", type=float, default=20.0)
    parser.add_argument("--versions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--max-offset", type=float, default=1.5)
    parser.add_argument(
        "--output-dir",
        default="outputs/ref_augmentation_audit"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    rows = load_rows(args.data)
    if not rows:
        raise ValueError("Dataset is empty")

    # Select one fixed scene for all augmentation versions.
    if args.scene_id:
        candidates = [
            r for r in rows if r["id"] == args.scene_id
        ]
    else:
        candidates = [
            r for r in rows
            if r["navigation"] == args.navigation
            and abs(
                float(r["metadata"]["ego_distance_to_junction_m"])
                - args.distance
            ) < 0.1
        ]

    if not candidates:
        raise ValueError(
            "No matching scene. Check --data, --scene-id, "
            "--navigation and --distance."
        )

    row = candidates[0]
    spacing = float(row.get("gt_path_sample_spacing_m", 1.0))
    gt = np.asarray(row["gt_path"], dtype=np.float64)

    ref = resample(
        row["reference_line"], len(gt), step=spacing
    ).astype(np.float64)
    ref -= ref[:1]

    # Reference-normal direction, used to measure lateral perturbation.
    tangent = np.gradient(ref, axis=0)
    tangent /= np.maximum(
        np.linalg.norm(tangent, axis=1, keepdims=True), 1e-8
    )
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)

    versions = []
    records = []

    def ade(path):
        return float(np.linalg.norm(path - gt, axis=1).mean())

    print("Scene:", row["id"])
    print("Navigation:", row["navigation"])
    print(
        "Distance:",
        row["metadata"]["ego_distance_to_junction_m"], "m"
    )
    print("GT points:", len(gt))
    print("Original Reference ADE:", f"{ade(ref):.4f} m")
    print()

    for i in range(args.versions):
        aug = augment_reference_geometry(
            ref,
            max_offset_m=args.max_offset,
        ).astype(np.float64)

        delta = aug - ref
        lateral = np.sum(delta * normal, axis=1)
        steps = np.linalg.norm(np.diff(aug, axis=0), axis=1)

        rec = {
            "version": i + 1,
            "lateral_rms_m": float(np.sqrt(np.mean(lateral**2))),
            "lateral_max_abs_m": float(np.max(np.abs(lateral))),
            "displacement_rms_m": float(
                np.sqrt(np.mean(np.sum(delta**2, axis=1)))
            ),
            "displacement_max_m": float(
                np.linalg.norm(delta, axis=1).max()
            ),
            "reference_ade_m": ade(aug),
            "ade_change_m": ade(aug) - ade(ref),
            "spacing_mean_m": float(steps.mean()),
            "spacing_mae_from_target_m": float(
                np.mean(np.abs(steps - spacing))
            ),
            "spacing_min_m": float(steps.min()),
            "spacing_max_m": float(steps.max()),
            "origin_error_m": float(np.linalg.norm(aug[0])),
        }

        versions.append(aug)
        records.append(rec)

        print(
            f"V{i+1:02d} | "
            f"lateral RMS={rec['lateral_rms_m']:.3f}m | "
            f"lateral max={rec['lateral_max_abs_m']:.3f}m | "
            f"Ref ADE={rec['reference_ade_m']:.3f}m | "
            f"spacing MAE={rec['spacing_mae_from_target_m']:.4f}m"
        )

    # Pairwise curve diversity across the 10 versions.
    pairwise = []
    for (i, a), (j, b) in combinations(enumerate(versions), 2):
        rms = float(np.sqrt(np.mean(np.sum((a - b)**2, axis=1))))
        pairwise.append({
            "version_a": i + 1,
            "version_b": j + 1,
            "curve_rms_difference_m": rms,
        })

    pair_dist = [p["curve_rms_difference_m"] for p in pairwise]

    print("\n===== Diversity summary =====")
    print(
        "Pairwise curve RMS difference (m):",
        f"mean={np.mean(pair_dist):.3f}",
        f"min={np.min(pair_dist):.3f}",
        f"max={np.max(pair_dist):.3f}",
    )
    print(
        "Lateral RMS range (m):",
        f"{min(r['lateral_rms_m'] for r in records):.3f}",
        "to",
        f"{max(r['lateral_rms_m'] for r in records):.3f}",
    )
    print(
        "Augmented Reference ADE range (m):",
        f"{min(r['reference_ade_m'] for r in records):.3f}",
        "to",
        f"{max(r['reference_ade_m'] for r in records):.3f}",
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "versions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    with (out / "pairwise.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=pairwise[0].keys()
        )
        writer.writeheader()
        writer.writerows(pairwise)

    # Save all curves so they can be plotted later.
    np.savez_compressed(
        out / "curves.npz",
        original_reference=ref,
        gt_path=gt,
        augmented_references=np.stack(versions),
    )

    print("\nSaved:", out)
    print("  versions.csv")
    print("  pairwise.csv")
    print("  curves.npz")


if __name__ == "__main__":
    main()

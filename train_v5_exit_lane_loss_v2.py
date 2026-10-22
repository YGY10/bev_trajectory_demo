#!/usr/bin/env python3
"""Unified V5/V6B1 training with visible target-exit-lane lateral supervision.

Examples
--------
# V5 model (legacy BEV encoder)
python train_v5_exit_lane_loss_v2.py --model-type v5

# V6B1 with CNN and pooled line tokens
python train_v5_exit_lane_loss_v2.py --model-type v6B1 --bev-encoder cnn --bev-token-mode pooled

# V6B1 with CNN and individual point tokens (memory intensive)
python train_v5_exit_lane_loss_v2.py --model-type v6B1 --bev-encoder cnn --bev-token-mode point --batch-size 1

The model inputs, path loss, exit-lane supervision, data split files, optimizer,
validation metrics, and checkpoint-selection rules are shared between versions.
Exit-lane geometry is supervision only; it is never passed into the model.
"""
import argparse
import importlib
import inspect
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import BEVDataset
from train import path_loss
from evaluate_exit_lane_v5 import base_geometry, observed_sides, near_visible


MODEL_IMPORTS = {
    "v5": ("models.trajectory_model", "arc_length_path_v5"),
    "v6B1": ("models.trajectory_model_v6B1", "arc_length_path_v6B1"),
    "v6B1G": ("models.trajectory_model_v6B1G", "arc_length_path_v6B1G"),
}
MODEL_INPUT_KEYS = ("lines", "line_valid", "reference", "navigation")


class ExitSupervisedDataset(BEVDataset):
    """V5-format dataset with lane geometry added only as training supervision."""

    def __init__(self, path, *, seed=20261021, max_lines=32,
                 visibility_gap=2.0, min_visible_points=5, augment=False, ref_augment=False):
        # train.jsonl and val.jsonl are already split files; do not split them again.
        super().__init__(path, split="all", augment=False, max_lines=max_lines)
        self.augment = bool(augment)
        self.ref_augment = bool(augment and ref_augment)
        self.exit_supervision = []
        self.count_eligible = 0

        for row in self.rows:
            b, v, n, width, _ = base_geometry(row, seed)
            sides = observed_sides(
                row, b, v, n, width, row["metadata"]["exit_lane_index"]
            )
            gt = np.asarray(row["gt_path"], dtype=np.float64)
            station = (gt - b) @ v
            mask = (
                (station >= 3.0)
                & near_visible(station, sides[0], visibility_gap)
                & near_visible(station, sides[1], visibility_gap)
            )
            if int(mask.sum()) < min_visible_points:
                mask[:] = False
            else:
                self.count_eligible += 1
                # Independent geometry check: do not infer the lane center from GT.
                deviation = np.abs((gt[mask] - b) @ n)
                if deviation.max() > 0.1:
                    raise ValueError(
                        f'{row["id"]}: exit geometry/GT mismatch '
                        f'({deviation.max():.3f}m). Check generator seed.'
                    )
            self.exit_supervision.append((
                torch.from_numpy(mask.copy()),
                torch.tensor(b, dtype=torch.float32),
                torch.tensor(n, dtype=torch.float32),
            ))

    def __getitem__(self, index):
        item = super().__getitem__(index)
        mask, b, n = self.exit_supervision[index]
        item["exit_mask"] = mask
        item["exit_anchor"] = b
        item["exit_normal"] = n
        return item


def exit_lateral_stats(pred, batch):
    """Scene-balanced absolute lateral error over strictly visible exit points."""
    mask = batch["exit_mask"].bool()
    lateral = (
        (pred - batch["exit_anchor"][:, None, :])
        * batch["exit_normal"][:, None, :]
    ).sum(-1).abs()
    counts = mask.sum(dim=1)
    valid = counts > 0
    sums = (lateral * mask.to(lateral.dtype)).sum(dim=1)
    per_scene = sums / counts.clamp_min(1)

    if valid.any().item():
        loss = per_scene[valid].mean()
    else:
        loss = pred.sum() * 0.0

    return (
        loss,
        float(sums.detach().sum().item()),
        int(counts.sum().item()),
        float(per_scene[valid].detach().sum().item()),
        int(valid.sum().item()),
    )


def build_model(model_type, horizon, bev_encoder, bev_token_mode):
    """Instantiate the requested model while validating its constructor API."""
    if model_type not in MODEL_IMPORTS:
        raise ValueError(f"Unsupported model type: {model_type}")

    module_name, _ = MODEL_IMPORTS[model_type]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Could not import {module_name!r} for --model-type {model_type}. "
            "Check that the corresponding model file exists."
        ) from exc

    model_cls = getattr(module, "TrajectoryModel", None)
    if model_cls is None:
        raise AttributeError(f"{module_name} does not define TrajectoryModel")

    signature = inspect.signature(model_cls.__init__)
    params = signature.parameters
    kwargs = {}
    if "horizon" in params:
        kwargs["horizon"] = horizon
    else:
        raise TypeError(
            f"{module_name}.TrajectoryModel.__init__ has no 'horizon' parameter; "
            "update build_model() to match this model's constructor."
        )

    if model_type in ("v6B1", "v6B1G"):
        if "bev_encoder" not in params:
            raise TypeError(
                f"{module_name}.TrajectoryModel.__init__ has no 'bev_encoder' "
                "parameter; expected the configurable V6B1 implementation."
            )
        if "bev_token_mode" not in params:
            raise TypeError(
                f"{module_name}.TrajectoryModel.__init__ must accept bev_token_mode"
            )
        kwargs["bev_encoder"] = bev_encoder
        kwargs["bev_token_mode"] = bev_token_mode

    model = model_cls(**kwargs)
    return model, model_cls.__name__


def checkpoint(path, model, train_data, args, epoch, ade, fde,
               scene_lat, point_lat, model_type, model_class, selection_kind):
    """Save weights and sufficient metadata to identify the exact experiment."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _, architecture_tag = MODEL_IMPORTS[model_type]
    effective_encoder = args.bev_encoder if model_type == "v6B1" else None
    effective_token_mode = args.bev_token_mode if model_type == "v6B1" else None
    torch.save({
        "model": model.state_dict(),
        # Keep the architecture tag convention used by the existing experiments.
        "model_type": architecture_tag,
        "model_variant": model_type,
        "model_class": model_class,
        "bev_encoder": effective_encoder,
        "bev_token_mode": effective_token_mode,
        "horizon": train_data.path_points,
        "spacing": train_data.spacing,
        "epoch": epoch,
        "seed": args.seed,
        "max_lines": args.max_lines,
        "best_val_path_mean": ade,
        "val_path_end": fde,
        "val_visible_exit_lat_scene": scene_lat,
        "val_visible_exit_lat_point": point_lat,
        "lambda_lane": args.lambda_lane,
        "generator_seed": args.generator_seed,
        "training_variant": "visible_exit_lateral_scene_balanced_v2",
        "checkpoint_selection": selection_kind,
        "checkpoint_policy": args.checkpoint_selection,
        "config": vars(args).copy(),
    }, path)


def default_output_prefix(model_type, bev_encoder, bev_token_mode, lambda_lane):
    # Keep V5's historical prefix for the default lambda=0.02; isolate V6B1 outputs.
    lane_tag = f"{int(round(lambda_lane * 100)):03d}"
    token_tag = "" if bev_token_mode == "pooled" else f"_{bev_token_mode}"
    if model_type == "v5":
        return f"checkpoints/v5_lane_v2_lambda{lane_tag}_unified"
    return f"checkpoints/v6B1_lane_v2_lambda{lane_tag}_{bev_encoder}{token_tag}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-type", choices=tuple(MODEL_IMPORTS), default="v6B1",
                   help="Model implementation to train (default: v6B1).")
    p.add_argument("--bev-encoder", choices=("cnn", "mlp"), default="cnn",
                   help="V6B1 BEV encoder; ignored by V5's native architecture.")
    p.add_argument("--train-data", default="data/v5_splits/train.jsonl")
    p.add_argument("--val-data", default="data/v5_splits/val.jsonl")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--generator-seed", type=int, default=20261021)
    p.add_argument("--max-lines", type=int, default=32)
    p.add_argument("--lambda-lane", type=float, default=0.02)
    p.add_argument("--visibility-gap", type=float, default=2.0)
    p.add_argument("--min-visible-points", type=int, default=5)
    p.add_argument("--output-prefix", default=None,
                   help="Checkpoint prefix; defaults to a model-specific path.")
    p.add_argument("--checkpoint-selection", choices=("both", "ade", "exit"),
                   default="both",
                   help="Save best-ADE and/or best scene-balanced exit-lateral checkpoint.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument(
        "--bev-token-mode",
        choices=("pooled", "point"),
        default="pooled",
        help="V6B1 BEV tokenization: pooled or point-level.",
    )
    p.add_argument(
        "--ref-augment",
        action="store_true",
        help="Enable online smooth Reference geometry augmentation",
    )
    a = p.parse_args()

    if a.model_type == "v5" and a.bev_token_mode != "pooled":
        p.error("--bev-token-mode point requires --model-type v6B1")
    if a.lambda_lane < 0:
        p.error("--lambda-lane must be nonnegative")
    if a.epochs < 1 or a.batch_size < 1:
        p.error("--epochs and --batch-size must be positive")
    if a.max_lines < 1 or a.min_visible_points < 1:
        p.error("--max-lines and --min-visible-points must be positive")

    output_prefix = a.output_prefix or default_output_prefix(
        a.model_type, a.bev_encoder, a.bev_token_mode, a.lambda_lane
    )
    # Record the resolved path, not None, in checkpoint config metadata.
    a.output_prefix = output_prefix
    ade_path = output_prefix + "_best_ade.pt"
    exit_path = output_prefix + "_best_exit_lat.pt"
    paths_to_write = []
    if a.checkpoint_selection in ("both", "ade"):
        paths_to_write.append(ade_path)
    if a.checkpoint_selection in ("both", "exit"):
        paths_to_write.append(exit_path)
    if len(set(paths_to_write)) != len(paths_to_write):
        raise ValueError("Checkpoint output paths are not unique")
    existing = [path for path in paths_to_write if Path(path).exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing checkpoint(s): " + ", ".join(existing)
            + ". Choose a new --output-prefix or move the old checkpoints first."
        )

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    train_data = ExitSupervisedDataset(
        a.train_data, seed=a.generator_seed, max_lines=a.max_lines,
        visibility_gap=a.visibility_gap,
        min_visible_points=a.min_visible_points, augment=True, ref_augment=a.ref_augment,
    )
    val_data = ExitSupervisedDataset(
        a.val_data, seed=a.generator_seed, max_lines=a.max_lines,
        visibility_gap=a.visibility_gap,
        min_visible_points=a.min_visible_points, augment=False, ref_augment=False,
    )
    if train_data.path_points != val_data.path_points or train_data.spacing != val_data.spacing:
        raise ValueError("Train/validation horizon or spacing mismatch")

    tr_loader = DataLoader(
        train_data, batch_size=a.batch_size, shuffle=True, num_workers=a.workers
    )
    va_loader = DataLoader(
        val_data, batch_size=a.batch_size, shuffle=False, num_workers=a.workers
    )

    model, model_class = build_model(a.model_type, train_data.path_points, a.bev_encoder, a.bev_token_mode)
    model = model.to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    best_ade = float("inf")
    best_exit = float("inf")

    print(
        f"model_type={a.model_type} model_class={model_class} "
        f"bev_encoder={a.bev_encoder if a.model_type in ('v6B1', 'v6B1G') else 'native'} "
        f"bev_token_mode={a.bev_token_mode if a.model_type in ('v6B1', 'v6B1G') else 'native'} "
        f"lambda_lane={a.lambda_lane} train={len(train_data)} val={len(val_data)} "
        f"eligible_train={train_data.count_eligible} eligible_val={val_data.count_eligible} "
        f"generator_seed={a.generator_seed} device={a.device}",
        flush=True,
    )
    print(f"Checkpoint prefix: {output_prefix}", flush=True)

    for epoch in range(1, a.epochs + 1):
        model.train()
        summed_loss = 0.0
        lateral_numer = 0.0
        npoints = 0
        train_scene_sum = 0.0
        train_scene_count = 0

        for batch in tr_loader:
            batch = {k: v.to(a.device) for k, v in batch.items()}
            model_inputs = {k: batch[k] for k in MODEL_INPUT_KEYS}
            pred = model(model_inputs)
            base, _, _ = path_loss(pred, batch["target_xy"], train_data.spacing)
            lane, ps, pn, ss, sn = exit_lateral_stats(pred, batch)
            loss = base + a.lambda_lane * lane

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            summed_loss += float(loss.item()) * pred.shape[0]
            lateral_numer += ps
            npoints += pn
            train_scene_sum += ss
            train_scene_count += sn

        model.eval()
        total_ade_dist = 0.0
        total_end_dist = 0.0
        total_spacing = 0.0
        total_items = 0
        lateral_sum = 0.0
        count = 0
        scene_sum = 0.0
        scene_count = 0

        with torch.inference_mode():
            for batch in va_loader:
                batch = {k: v.to(a.device) for k, v in batch.items()}
                model_inputs = {k: batch[k] for k in MODEL_INPUT_KEYS}
                pred = model(model_inputs)
                d = torch.linalg.vector_norm(pred - batch["target_xy"], dim=-1)
                seg = (
                    torch.linalg.vector_norm(pred[:, 1:] - pred[:, :-1], dim=-1)
                    - train_data.spacing
                ).abs()
                _, ps, pn, ss, sn = exit_lateral_stats(pred, batch)
                total_ade_dist += d.sum().item()
                total_end_dist += d[:, -1].sum().item()
                total_spacing += seg.sum().item()
                total_items += pred.shape[0]
                lateral_sum += ps
                count += pn
                scene_sum += ss
                scene_count += sn

        ade = total_ade_dist / (total_items * train_data.path_points)
        fde = total_end_dist / total_items
        spacing_err = total_spacing / (total_items * (train_data.path_points - 1))
        exit_lat_point = lateral_sum / max(count, 1)
        exit_lat_scene = scene_sum / max(scene_count, 1)
        if scene_count == 0:
            raise RuntimeError(
                "No eligible validation scenes; check data and geometry masks"
            )

        print(
            f"Epoch {epoch:03d}: train_loss={summed_loss / len(train_data):.4f} "
            f"train_exit_point={lateral_numer / max(npoints, 1):.3f}m "
            f"train_exit_scene={train_scene_sum / max(train_scene_count, 1):.3f}m "
            f"val_path_mean={ade:.3f}m val_path_end={fde:.3f}m "
            f"val_exit_point={exit_lat_point:.3f}m "
            f"val_exit_scene={exit_lat_scene:.3f}m "
            f"val_spacing_err={spacing_err:.3f}m",
            flush=True,
        )

        if a.checkpoint_selection in ("both", "ade") and ade < best_ade:
            best_ade = ade
            checkpoint(
                ade_path, model, train_data, a, epoch, ade, fde,
                exit_lat_scene, exit_lat_point, a.model_type, model_class, "best_ade",
            )
        if (a.checkpoint_selection in ("both", "exit")
                and exit_lat_scene < best_exit):
            best_exit = exit_lat_scene
            checkpoint(
                exit_path, model, train_data, a, epoch, ade, fde,
                exit_lat_scene, exit_lat_point, a.model_type, model_class,
                "best_exit_lat_scene",
            )

    if a.checkpoint_selection in ("both", "ade"):
        print("Saved best-ADE checkpoint:", ade_path, flush=True)
    if a.checkpoint_selection in ("both", "exit"):
        print("Saved best scene-balanced exit checkpoint:", exit_path, flush=True)


if __name__ == "__main__":
    main()
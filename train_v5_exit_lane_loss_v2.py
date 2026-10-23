#!/usr/bin/env python3
"""V5/V6B1/V6B1G path training with strictly visible exit-lane supervision.

Keeps the original training loss and validation metric definitions, and adds:
  * navigation-stratified strict exit-lateral validation/reference baselines;
  * a global (ADE, scene-balanced exit lateral) Pareto frontier;
  * a constrained best-ADE checkpoint, requiring each navigation's exit
    lateral error <= its Reference baseline + --exit-constraint-tolerance;
  * complete CSV training history and Pareto JSON manifest.

The generator's target-lane geometry is used for loss/evaluation ONLY.
It is never supplied to the model as an input feature.
"""
import argparse
import csv
import importlib
import inspect
import json
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
NAV_NAMES = ("straight", "left", "right")
NAV_TO_ID = {name: idx for idx, name in enumerate(NAV_NAMES)}


class ExitSupervisedDataset(BEVDataset):
    """Dataset with independently reconstructed, strictly visible exit supervision."""

    def __init__(self, path, *, seed=20261021, max_lines=32,
                 visibility_gap=2.0, min_visible_points=5,
                 augment=False, ref_augment=False):
        # The supplied train.jsonl and val.jsonl are already disjoint files.
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


def per_scene_exit_lateral(pred, batch):
    """Return lateral mean per scene, validity mask and masked point sums."""
    mask = batch["exit_mask"].bool()
    lateral = (
        (pred - batch["exit_anchor"][:, None, :])
        * batch["exit_normal"][:, None, :]
    ).sum(-1).abs()
    counts = mask.sum(dim=1)
    sums = (lateral * mask.to(lateral.dtype)).sum(dim=1)
    per_scene = sums / counts.clamp_min(1)
    return per_scene, counts > 0, sums, counts


def exit_lateral_stats(pred, batch):
    """Original loss/statistics interface, including batches without exit GT."""
    per_scene, valid, sums, counts = per_scene_exit_lateral(pred, batch)
    loss = per_scene[valid].mean() if valid.any().item() else pred.sum() * 0.0
    return (
        loss,
        float(sums.detach().sum().item()),
        int(counts.sum().item()),
        float(per_scene[valid].detach().sum().item()),
        int(valid.sum().item()),
    )


def build_model(model_type, horizon, bev_encoder, bev_token_mode):
    if model_type not in MODEL_IMPORTS:
        raise ValueError(f"Unsupported model type: {model_type}")
    module_name, _ = MODEL_IMPORTS[model_type]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(f"Could not import {module_name!r}") from exc
    model_cls = getattr(module, "TrajectoryModel", None)
    if model_cls is None:
        raise AttributeError(f"{module_name} does not define TrajectoryModel")
    params = inspect.signature(model_cls.__init__).parameters
    if "horizon" not in params:
        raise TypeError(f"{module_name}.TrajectoryModel has no horizon parameter")
    kwargs = {"horizon": horizon}
    if model_type in ("v6B1", "v6B1G"):
        if "bev_encoder" not in params or "bev_token_mode" not in params:
            raise TypeError(f"{module_name}.TrajectoryModel lacks BEV configuration")
        kwargs.update(bev_encoder=bev_encoder, bev_token_mode=bev_token_mode)
    return model_cls(**kwargs), model_cls.__name__


def checkpoint(path, model, train_data, args, epoch, ade, fde,
               scene_lat, point_lat, model_type, model_class, selection_kind,
               extra_metrics=None):
    """Save checkpoint in the existing schema, extended with selection metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _, architecture_tag = MODEL_IMPORTS[model_type]
    is_v6 = model_type in ("v6B1", "v6B1G")
    payload = {
        "model": model.state_dict(),
        "model_type": architecture_tag,
        "model_variant": model_type,
        "model_class": model_class,
        "bev_encoder": args.bev_encoder if is_v6 else None,
        "bev_token_mode": args.bev_token_mode if is_v6 else None,
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
        "extra_metrics": extra_metrics or {},
        "config": vars(args).copy(),
    }
    torch.save(payload, path)


def default_output_prefix(model_type, bev_encoder, bev_token_mode, lambda_lane):
    lane_tag = f"{int(round(lambda_lane * 100)):03d}"
    token_tag = "" if bev_token_mode == "pooled" else f"_{bev_token_mode}"
    if model_type == "v5":
        return f"checkpoints/v5_lane_v2_lambda{lane_tag}_unified"
    return f"checkpoints/{model_type}_lane_v2_lambda{lane_tag}_{bev_encoder}{token_tag}"


def aggregate_nav_lateral(loader, device=None, model=None):
    """Macro-average strict exit lateral errors over eligible scenes by nav.

    If model is None, evaluate the unaugmented Reference directly.
    """
    totals = {name: 0.0 for name in NAV_NAMES}
    counts = {name: 0 for name in NAV_NAMES}
    for batch in loader:
        if model is None:
            pred = batch["reference"]
        else:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model({k: batch[k] for k in MODEL_INPUT_KEYS})
        per_scene, valid, _, _ = per_scene_exit_lateral(pred, batch)
        for nav, nav_id in NAV_TO_ID.items():
            selected = valid & (batch["navigation"] == nav_id)
            totals[nav] += float(per_scene[selected].sum().item())
            counts[nav] += int(selected.sum().item())
    values = {
        nav: totals[nav] / counts[nav] if counts[nav] else float("nan")
        for nav in NAV_NAMES
    }
    return values, counts


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def dominates(ade_a, lat_a, ade_b, lat_b):
    """A dominates B if A is no worse in both and better in at least one."""
    return (ade_a <= ade_b and lat_a <= lat_b
            and (ade_a < ade_b or lat_a < lat_b))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-type", choices=tuple(MODEL_IMPORTS), default="v6B1")
    p.add_argument("--bev-encoder", choices=("cnn", "mlp"), default="cnn")
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
    p.add_argument("--output-prefix", default=None)
    p.add_argument(
        "--checkpoint-selection", choices=("both", "ade", "exit", "pareto"),
        default="both", help="pareto also saves best ADE, best exit and best feasible",
    )
    p.add_argument(
        "--exit-constraint-tolerance", type=float, default=0.0,
        help="Allowed per-navigation degradation from Reference exit lateral error (m)",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--bev-token-mode", choices=("pooled", "point"), default="pooled")
    p.add_argument("--ref-augment", action="store_true")
    a = p.parse_args()

    if a.model_type == "v5" and a.bev_token_mode != "pooled":
        p.error("--bev-token-mode point requires a V6 model")
    if a.model_type == "v6B1G" and a.bev_token_mode != "point":
        p.error("--model-type v6B1G requires --bev-token-mode point")
    if a.lambda_lane < 0 or a.exit_constraint_tolerance < 0:
        p.error("lambda_lane and exit_constraint_tolerance must be nonnegative")
    if a.epochs < 1 or a.batch_size < 1:
        p.error("--epochs and --batch-size must be positive")
    if a.max_lines < 1 or a.min_visible_points < 1:
        p.error("--max-lines and --min-visible-points must be positive")

    output_prefix = a.output_prefix or default_output_prefix(
        a.model_type, a.bev_encoder, a.bev_token_mode, a.lambda_lane
    )
    a.output_prefix = output_prefix
    ade_path = Path(output_prefix + "_best_ade.pt")
    exit_path = Path(output_prefix + "_best_exit_lat.pt")
    feasible_path = Path(output_prefix + "_best_feasible.pt")
    history_path = Path(output_prefix + "_history.csv")
    pareto_manifest = Path(output_prefix + "_pareto.json")
    pareto_dir = Path(output_prefix + "_pareto")
    reference_path = Path(output_prefix + "_reference_exit_nav.json")

    save_ade = a.checkpoint_selection in ("both", "ade", "pareto")
    save_exit = a.checkpoint_selection in ("both", "exit", "pareto")
    use_pareto = a.checkpoint_selection == "pareto"
    planned_files = [history_path, reference_path]
    if save_ade:
        planned_files.append(ade_path)
    if save_exit:
        planned_files.append(exit_path)
    if use_pareto:
        planned_files.extend([feasible_path, pareto_manifest])
    existing = [str(path) for path in planned_files if path.exists()]
    if use_pareto and pareto_dir.exists():
        existing.append(str(pareto_dir))
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing outputs: " + ", ".join(existing)
            + ". Choose a new --output-prefix."
        )
    if len(set(planned_files)) != len(planned_files):
        raise ValueError("Output paths collide")

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    train_data = ExitSupervisedDataset(
        a.train_data, seed=a.generator_seed, max_lines=a.max_lines,
        visibility_gap=a.visibility_gap, min_visible_points=a.min_visible_points,
        augment=True, ref_augment=a.ref_augment,
    )
    val_data = ExitSupervisedDataset(
        a.val_data, seed=a.generator_seed, max_lines=a.max_lines,
        visibility_gap=a.visibility_gap, min_visible_points=a.min_visible_points,
        augment=False, ref_augment=False,
    )
    if (train_data.path_points != val_data.path_points
            or train_data.spacing != val_data.spacing):
        raise ValueError("Train/validation horizon or spacing mismatch")

    tr_loader = DataLoader(
        train_data, batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
    )
    va_loader = DataLoader(
        val_data, batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
    )
    with torch.inference_mode():
        ref_exit_nav, ref_nav_count = aggregate_nav_lateral(va_loader)
    # A navigation with zero evaluable scenes cannot satisfy the requested
    # per-navigation constraint. Never silently drop it.
    if use_pareto and any(ref_nav_count[nav] == 0 for nav in NAV_NAMES):
        raise RuntimeError(
            "Cannot apply per-navigation constraints because validation has "
            f"no strictly eligible scenes for: "
            f"{[nav for nav in NAV_NAMES if ref_nav_count[nav] == 0]}"
        )
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_text(json.dumps({
        "reference_exit_nav": ref_exit_nav,
        "strict_eligible_scenes_by_nav": ref_nav_count,
        "generator_seed": a.generator_seed,
        "visibility_gap": a.visibility_gap,
        "min_visible_points": a.min_visible_points,
        "constraint_tolerance": a.exit_constraint_tolerance,
    }, indent=2), encoding="utf-8")

    model, model_class = build_model(
        a.model_type, train_data.path_points, a.bev_encoder, a.bev_token_mode
    )
    model = model.to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    best_ade = float("inf")
    best_exit = float("inf")
    best_feasible_ade = float("inf")
    pareto_front = []
    history = []
    if use_pareto:
        pareto_dir.mkdir(parents=True, exist_ok=False)

    print(
        f"model_type={a.model_type} model_class={model_class} "
        f"bev_encoder={a.bev_encoder if a.model_type != 'v5' else 'native'} "
        f"bev_token_mode={a.bev_token_mode if a.model_type != 'v5' else 'native'} "
        f"lambda_lane={a.lambda_lane} train={len(train_data)} val={len(val_data)} "
        f"eligible_train={train_data.count_eligible} "
        f"eligible_val={val_data.count_eligible} "
        f"generator_seed={a.generator_seed} device={a.device}",
        flush=True,
    )
    print("Reference strict exit lat by nav:", ref_exit_nav, flush=True)
    print("Checkpoint prefix:", output_prefix, flush=True)

    for epoch in range(1, a.epochs + 1):
        model.train()
        summed_loss = 0.0
        lateral_numer = 0.0
        npoints = 0
        train_scene_sum = 0.0
        train_scene_count = 0
        train_base_sum = 0.0
        train_weighted_exit_sum = 0.0

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

            size = pred.shape[0]
            summed_loss += float(loss.item()) * size
            train_base_sum += float(base.item()) * size
            train_weighted_exit_sum += float((a.lambda_lane * lane).item()) * size
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
        point_count = 0
        scene_sum = 0.0
        scene_count = 0
        nav_sum = {name: 0.0 for name in NAV_NAMES}
        nav_count = {name: 0 for name in NAV_NAMES}

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
                per_scene, valid, _, _ = per_scene_exit_lateral(pred, batch)

                total_ade_dist += d.sum().item()
                total_end_dist += d[:, -1].sum().item()
                total_spacing += seg.sum().item()
                total_items += pred.shape[0]
                lateral_sum += ps
                point_count += pn
                scene_sum += ss
                scene_count += sn
                for nav, nav_id in NAV_TO_ID.items():
                    selected = valid & (batch["navigation"] == nav_id)
                    nav_sum[nav] += float(per_scene[selected].sum().item())
                    nav_count[nav] += int(selected.sum().item())

        if not total_items or scene_count == 0:
            raise RuntimeError("No eligible validation scenes; check geometry masks")
        ade = total_ade_dist / (total_items * train_data.path_points)
        fde = total_end_dist / total_items
        spacing_err = total_spacing / (total_items * (train_data.path_points - 1))
        exit_lat_point = lateral_sum / max(point_count, 1)
        exit_lat_scene = scene_sum / scene_count
        val_exit_nav = {
            nav: nav_sum[nav] / nav_count[nav] if nav_count[nav] else float("nan")
            for nav in NAV_NAMES
        }
        feasible = all(
            np.isfinite(val_exit_nav[nav])
            and val_exit_nav[nav] <= ref_exit_nav[nav] + a.exit_constraint_tolerance
            for nav in NAV_NAMES
        )
        extra_metrics = {
            "reference_exit_nav": dict(ref_exit_nav),
            "model_exit_nav": dict(val_exit_nav),
            "strict_eligible_scenes_by_nav": dict(nav_count),
            "exit_constraint_feasible": bool(feasible),
            "exit_constraint_tolerance": a.exit_constraint_tolerance,
            "val_spacing_err": spacing_err,
        }
        print(
            f"Epoch {epoch:03d}: train_loss={summed_loss / len(train_data):.4f} "
            f"train_base={train_base_sum / len(train_data):.4f} "
            f"train_weighted_exit={train_weighted_exit_sum / len(train_data):.4f} "
            f"train_exit_point={lateral_numer / max(npoints, 1):.3f}m "
            f"train_exit_scene={train_scene_sum / max(train_scene_count, 1):.3f}m "
            f"val_path_mean={ade:.3f}m val_path_end={fde:.3f}m "
            f"val_exit_point={exit_lat_point:.3f}m "
            f"val_exit_scene={exit_lat_scene:.3f}m "
            f"val_spacing_err={spacing_err:.3f}m "
            f"exit_by_nav=" + ",".join(
                f"{nav}:{val_exit_nav[nav]:.4f}(ref:{ref_exit_nav[nav]:.4f})"
                for nav in NAV_NAMES
            ) + f" feasible={feasible}",
            flush=True,
        )

        if save_ade and ade < best_ade:
            best_ade = ade
            checkpoint(
                ade_path, model, train_data, a, epoch, ade, fde,
                exit_lat_scene, exit_lat_point, a.model_type, model_class,
                "best_ade", extra_metrics,
            )
        if save_exit and exit_lat_scene < best_exit:
            best_exit = exit_lat_scene
            checkpoint(
                exit_path, model, train_data, a, epoch, ade, fde,
                exit_lat_scene, exit_lat_point, a.model_type, model_class,
                "best_exit_lat_scene", extra_metrics,
            )

        if use_pareto:
            # Equal metrics: retain the older checkpoint; avoid duplicated points.
            same = any(
                old["ade"] == ade and old["exit_lat"] == exit_lat_scene
                for old in pareto_front
            )
            dominated = same or any(
                dominates(old["ade"], old["exit_lat"], ade, exit_lat_scene)
                for old in pareto_front
            )
            if not dominated:
                new_path = pareto_dir / f"epoch_{epoch:03d}.pt"
                # Save the new candidate first. Delete dominated older files
                # only after the new checkpoint write has succeeded.
                checkpoint(
                    new_path, model, train_data, a, epoch, ade, fde,
                    exit_lat_scene, exit_lat_point, a.model_type, model_class,
                    "pareto", extra_metrics,
                )
                survivors = []
                for old in pareto_front:
                    if dominates(ade, exit_lat_scene, old["ade"], old["exit_lat"]):
                        Path(old["path"]).unlink()
                    else:
                        survivors.append(old)
                pareto_front = survivors + [{
                    "epoch": epoch,
                    "ade": ade,
                    "exit_lat": exit_lat_scene,
                    "path": str(new_path),
                    "feasible": bool(feasible),
                    "model_exit_nav": dict(val_exit_nav),
                }]
            if feasible and ade < best_feasible_ade:
                best_feasible_ade = ade
                checkpoint(
                    feasible_path, model, train_data, a, epoch, ade, fde,
                    exit_lat_scene, exit_lat_point, a.model_type, model_class,
                    "best_feasible", extra_metrics,
                )

        history.append({
            "epoch": epoch,
            "train_loss": summed_loss / len(train_data),
            "train_base": train_base_sum / len(train_data),
            "train_weighted_exit": train_weighted_exit_sum / len(train_data),
            "val_ade": ade,
            "val_fde": fde,
            "val_spacing_err": spacing_err,
            "exit_lat_scene": exit_lat_scene,
            "exit_lat_point": exit_lat_point,
            "feasible": bool(feasible),
            **{f"exit_{nav}": val_exit_nav[nav] for nav in NAV_NAMES},
            **{f"reference_exit_{nav}": ref_exit_nav[nav] for nav in NAV_NAMES},
        })
        # Save history/manifest every epoch so an interrupted long experiment
        # still has auditable validation metrics and current Pareto paths.
        write_csv(history_path, history)
        if use_pareto:
            pareto_manifest.write_text(json.dumps({
                "objectives": ["val_ade", "exit_lat_scene"],
                "reference_exit_nav": ref_exit_nav,
                "exit_constraint_tolerance": a.exit_constraint_tolerance,
                "best_feasible_ade": (
                    best_feasible_ade if np.isfinite(best_feasible_ade) else None
                ),
                "frontier": sorted(pareto_front, key=lambda r: r["ade"]),
            }, indent=2), encoding="utf-8")

    if save_ade:
        print("Saved best-ADE checkpoint:", ade_path, flush=True)
    if save_exit:
        print("Saved best scene-balanced exit checkpoint:", exit_path, flush=True)
    if use_pareto:
        print("Pareto frontier checkpoints:", len(pareto_front), flush=True)
        if np.isfinite(best_feasible_ade):
            print("Saved best feasible checkpoint:", feasible_path,
                  "ADE:", f"{best_feasible_ade:.4f}", flush=True)
        else:
            print(
                "NO FEASIBLE CHECKPOINT: per-navigation strict exit constraints "
                "were never jointly satisfied. No best_feasible.pt was saved.",
                flush=True,
            )
        print("Pareto manifest:", pareto_manifest, flush=True)
    print("Training history:", history_path, flush=True)


if __name__ == "__main__":
    main()
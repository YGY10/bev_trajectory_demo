"""Checkpoint-aware loader for the V5 and V6B1 path-prediction models.

Run scripts from the repository root so the project's `models` package is importable.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import torch

MODEL_MODULES = {
    "v5": "models.trajectory_model",
    "v6B1": "models.trajectory_model_v6B1",
    "v6B1G": "models.trajectory_model_v6B1G",
}


def _infer_variant(checkpoint, state_dict):
    variant = checkpoint.get("model_variant")
    if variant in MODEL_MODULES:
        return variant

    tag = str(checkpoint.get("model_type", "")).lower()
    keys = tuple(state_dict.keys())
    if "v6b1" in tag or any(k.startswith("point_cnn.") for k in keys):
        return "v6B1"
    # Legacy V5 checkpoints predate model_variant metadata. V6B1's MLP mode
    # has the same point_mlp key pattern, so the explicit model_type tag is
    # preferred when available; otherwise the legacy checkpoint is treated as V5.
    return "v5"


def load_trajectory_model(checkpoint_path, device="cpu"):
    """Load a V5 or V6B1 checkpoint and return (model, checkpoint, metadata)."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(
            f"{path} is not a supported training checkpoint: expected a dict with 'model'."
        )
    state_dict = checkpoint["model"]
    if not isinstance(state_dict, dict):
        raise ValueError(f"Invalid state_dict in checkpoint: {path}")

    variant = _infer_variant(checkpoint, state_dict)
    if variant not in MODEL_MODULES:
        raise ValueError(f"Unsupported model variant {variant!r} in {path}")

    if variant == "v6B1":
        encoder = checkpoint.get("bev_encoder")
        if encoder not in ("cnn", "mlp"):
            encoder = "cnn" if any(k.startswith("point_cnn.") for k in state_dict) else "mlp"
    else:
        encoder = "native"

    module = importlib.import_module(MODEL_MODULES[variant])
    model_cls = getattr(module, "TrajectoryModel", None)
    if model_cls is None:
        raise AttributeError(f"{MODEL_MODULES[variant]} does not define TrajectoryModel")

    kwargs = {"horizon": int(checkpoint.get("horizon", 81))}

    if variant in ("v6B1", "v6B1G"):
        kwargs = {
    "horizon": int(checkpoint.get("horizon", 81))
}

    if variant in ("v6B1", "v6B1G"):
        encoder = checkpoint.get("bev_encoder")

        if encoder not in ("cnn", "mlp"):
            encoder = (
                "cnn"
                if any(k.startswith("point_cnn.") for k in state_dict)
                else "mlp"
            )

        token_mode = checkpoint.get("bev_token_mode")

        if token_mode not in ("pooled", "point"):
            token_mode = (
                "pooled"
                if any(k.startswith("line_mlp.") for k in state_dict)
                else "point"
            )

        kwargs["bev_encoder"] = encoder
        kwargs["bev_token_mode"] = token_mode


    model = model_cls(**kwargs)
    # Strict loading is intentional: it catches a mismatched model/checkpoint
    # rather than silently running an invalid sensitivity experiment.
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    info = {
        "model_variant": variant,
        "bev_encoder": encoder,
        "bev_token_mode": kwargs.get("bev_token_mode", "native"),
        "model_module": MODEL_MODULES[variant],
        "horizon": int(checkpoint.get("horizon", 81)),
        "max_lines": int(checkpoint.get("max_lines", 32)),
    }
    return model, checkpoint, info


#!/usr/bin/env python3
"""V6B1-G geometry-attention ablation; no training or checkpoint changes."""

import argparse
import torch
import torch.nn.functional as F

from datasets import BEVDataset
from tools.model_compat import load_trajectory_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/v6B1G_point_GeoAttn_5ep_best_ade.pt",
    )
    parser.add_argument("--data", default="data/v5_splits/val.jsonl")
    parser.add_argument("--navigation",
                        choices=["straight", "left", "right", "all"],
                        default="straight")
    parser.add_argument("--max-lines", type=int, default=32)
    parser.add_argument("--delta-m", type=float, default=0.5)
    parser.add_argument("--queries", default="20,50,70,80")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model, ckpt, info = load_trajectory_model(
        args.checkpoint, device
    )
    if not hasattr(model, "geometry_attn"):
        raise RuntimeError("Checkpoint model has no geometry_attn")

    model.eval()
    ds = BEVDataset(
        args.data, split="all", augment=False,
        max_lines=args.max_lines,
    )

    nav_names = {0: "straight", 1: "left", 2: "right"}
    candidates = [
        i for i, row in enumerate(ds.rows)
        if args.navigation == "all"
        or row["navigation"] == args.navigation
    ]
    if not candidates:
        raise RuntimeError("No matching scenes")

    # Pick a deterministic representative scene.
    idx = candidates[0]
    sample = ds[idx]
    batch = {
        k: sample[k].unsqueeze(0).to(device)
        for k in ("lines", "line_valid", "reference", "navigation")
    }

    captured = {}

    def capture_inputs(module, inputs):
        names = ("q", "bev", "ref", "xy", "direction", "valid")
        captured.update({
            name: value.detach().clone()
            for name, value in zip(names, inputs)
        })

    handle = model.geometry_attn.register_forward_pre_hook(
        capture_inputs
    )
    with torch.no_grad():
        pred_baseline = model(batch).detach().clone()
    handle.remove()

    attn = model.geometry_attn
    q = captured["q"]
    bev = captured["bev"]
    ref = captured["ref"]
    xy = captured["xy"]
    direction = captured["direction"]
    valid = captured["valid"].bool()

    B, Q, D = q.shape
    S = bev.shape[1]
    H = attn.heads
    Dh = attn.head_dim

    if not valid.any(dim=1).all():
        raise RuntimeError("No valid BEV points in selected scene")

    tangent = torch.zeros_like(ref)
    tangent[:, 1:] = ref[:, 1:] - ref[:, :-1]
    tangent[:, 0] = tangent[:, 1]
    tangent = F.normalize(tangent, dim=-1, eps=1e-6)
    normal = torch.stack([-tangent[..., 1], tangent[..., 0]], -1)

    qh = attn.q_proj(q).reshape(B, Q, H, Dh).transpose(1, 2)
    kh = attn.k_proj(bev).reshape(B, S, H, Dh).transpose(1, 2)
    vh = attn.v_proj(bev).reshape(B, S, H, Dh).transpose(1, 2)

    # Keep Content Logits and Value vectors fixed across interventions.
    content = (qh @ kh.transpose(-2, -1)) * (Dh ** -0.5)

    def evaluate_geometry(xy_input, content_scale=1.0,
                          bias_scale=1.0):
        delta = xy_input[:, None] - ref[:, :, None]
        ds = (delta * tangent[:, :, None]).sum(-1)
        dl = (delta * normal[:, :, None]).sum(-1)

        radial = torch.sqrt(ds.square() + dl.square() + 1e-6)
        point_dir = F.normalize(direction, dim=-1, eps=1e-6)
        align = (
            point_dir[:, None] * tangent[:, :, None]
        ).sum(-1).abs()

        geometry = torch.stack([
            (ds / 10).clamp(-12, 12),
            (dl / 10).clamp(-12, 12),
            (radial / 10).clamp(0, 16),
            align,
            (radial / 10).square().clamp(0, 256),
        ], dim=-1)

        bias = attn.bias_mlp(geometry).permute(0, 3, 1, 2)

        logits = content_scale * content + bias_scale * bias
        logits = logits.masked_fill(
            ~valid[:, None, None, :],
            torch.finfo(logits.dtype).min,
        )
        weights = torch.softmax(logits, dim=-1)

        attended = (weights @ vh).transpose(1, 2).reshape(B, Q, D)
        branch = (
            torch.sigmoid(attn.gate_logit)
            * attn.out_proj(attended)
        )

        return {
            "weights": weights,
            "attended": attended,
            "branch": branch,
            "bias": bias,
            "logits": logits,
            "radial": radial,
        }

    # A: Original attention.
    A = evaluate_geometry(xy, 1.0, 1.0)

    # B: Content-only attention; geometry bias removed.
    B0 = evaluate_geometry(xy, 1.0, 0.0)

    # C: Geometry-only attention; content logits removed.
    C = evaluate_geometry(xy, 0.0, 1.0)

    # D: Move geometry metadata only. Q/K/V and content logits stay fixed.
    xy_shifted = xy.clone()
    xy_shifted[..., 1] += args.delta_m
    D0 = evaluate_geometry(xy_shifted, 1.0, 1.0)

    def mean_l2(x):
        return x.norm(dim=-1).mean().item()

    def rms(x):
        return x.square().mean().sqrt().item()

    def tv(w1, w2):
        return 0.5 * (w1 - w2).abs().sum(dim=-1).mean().item()

    # Attention ignores a constant added to every valid-key logit for a given
    # query/head. Center over valid BEV keys so padded tokens do not affect the
    # diagnostics and common offsets are removed.
    valid_mask = valid[:, None, None, :]

    def center_over_valid_keys(x):
        mask = valid_mask.expand_as(x)
        safe_x = x.masked_fill(~mask, 0.0)
        count = mask.sum(dim=-1, keepdim=True).clamp_min(1)
        mean = safe_x.sum(dim=-1, keepdim=True) / count
        return (x - mean).masked_fill(~mask, 0.0)

    def masked_rms(x):
        return x.masked_select(valid_mask.expand_as(x)).square().mean().sqrt().item()

    def per_query_masked_rms(x):
        # x: [B,H,Q,S] -> one RMS per query, averaging batch/head/key
        mask = valid_mask.expand_as(x)
        sq = x.square().masked_fill(~mask, 0.0).sum(dim=(0, 1, 3))
        count = mask.expand_as(x).sum(dim=(0, 1, 3)).clamp_min(1)
        return (sq / count).sqrt()

    def per_query_tv(w1, w2):
        return 0.5 * (w1 - w2).abs().sum(dim=-1).mean(dim=(0, 1))

    def per_query_l2(x):
        # x: [B,Q,D] -> [Q]
        return x.norm(dim=-1).mean(dim=0)

    def print_condition(label, result):
        w = result["weights"][0]      # [H,Q,S]
        r = result["radial"][0]       # [Q,S]
        v = valid[0]                  # [S]
        nearest = r.masked_fill(~v[None, :], float("inf")).min(-1).values

        near5 = (w * ((r <= 5) & v)[None, :, :]).sum(-1).mean(0)
        near10 = (w * ((r <= 10) & v)[None, :, :]).sum(-1).mean(0)
        mean_dist = (w * r[None, :, :]).sum(-1).mean(0)

        p = w.clamp_min(1e-12)
        entropy = -(w * p.log()).sum(-1).mean(0)
        n_valid = int(v.sum())
        entropy_norm = entropy / max(torch.log(torch.tensor(
            float(n_valid), device=device
        )).item(), 1e-12)

        print(f"\n[{label}]")
        print("query | nearest_m | mass_5m | mass_10m | mean_dist_m | norm_entropy")
        for qi in query_indices:
            if qi >= Q:
                continue
            print(
                f"{qi:5d} | {nearest[qi].item():9.3f} | "
                f"{near5[qi].item():7.4f} | "
                f"{near10[qi].item():8.4f} | "
                f"{mean_dist[qi].item():11.3f} | "
                f"{entropy_norm[qi].item():12.4f}"
            )

    query_indices = [int(x) for x in args.queries.split(",")]

    print("Scene:", ds.rows[idx].get("id", idx))
    print("Navigation:", nav_names[int(sample["navigation"])])
    print("Model:", info["model_variant"])
    print("Valid BEV points:", int(valid.sum()))
    print("Gate:", torch.sigmoid(attn.gate_logit).item())
    print("Content logits std:", content.masked_select(
        valid[:, None, None, :].expand_as(content)
    ).std().item())
    print("Geometry bias std (raw, valid keys):", A["bias"].masked_select(
        valid[:, None, None, :].expand_as(A["bias"])
    ).std().item())
    content_centered = center_over_valid_keys(content)
    bias_centered = center_over_valid_keys(A["bias"])
    logits_centered = center_over_valid_keys(A["logits"])
    print("Content logits RMS (centered, valid keys):", masked_rms(content_centered))
    print("Geometry bias RMS (centered, valid keys):", masked_rms(bias_centered))
    print("Combined logits RMS (centered, valid keys):", masked_rms(logits_centered))

    print_condition("A: content + geometry", A)
    print_condition("B: content only", B0)
    print_condition("C: geometry only", C)
    print_condition("D: geometry shifted", D0)

    print("\n===== Pairwise intervention effects =====")
    for label, result in [
        ("A vs B", B0),
        ("A vs C", C),
        ("A vs D", D0),
    ]:
        print(f"\n{label}:")
        print("  Attention TV:", tv(A["weights"], result["weights"]))
        print("  Attended mean L2 change:",
              mean_l2(A["attended"] - result["attended"]))
        print("  Branch mean L2 change:",
              mean_l2(A["branch"] - result["branch"]))

    print("\n===== Geometry perturbation =====")
    print("Delta Y (m):", args.delta_m)
    raw_bias_delta = D0["bias"] - A["bias"]
    centered_bias_delta = center_over_valid_keys(D0["bias"]) - center_over_valid_keys(A["bias"])
    centered_logits_delta = center_over_valid_keys(D0["logits"]) - center_over_valid_keys(A["logits"])
    print("Raw bias delta RMS (valid keys):", masked_rms(raw_bias_delta))
    print("Centered bias delta RMS (valid keys):", masked_rms(centered_bias_delta))
    print("Centered logits delta RMS (valid keys):", masked_rms(centered_logits_delta))
    content_scale = max(masked_rms(content_centered), 1e-12)
    logits_scale = max(masked_rms(logits_centered), 1e-12)
    print("Centered bias delta / content-logit RMS:", masked_rms(centered_bias_delta) / content_scale)
    print("Centered logits delta / baseline-logit RMS:", masked_rms(centered_logits_delta) / logits_scale)
    print("Attention TV:", tv(A["weights"], D0["weights"]))
    print("Attended RMS change:", rms(D0["attended"] - A["attended"]))
    print("Branch RMS change:", rms(D0["branch"] - A["branch"]))

    print("\nPer-query geometry perturbation:")
    print("query | centered_bias_delta_rms | centered_logits_delta_rms | attention_TV | attended_delta_L2 | branch_delta_L2")
    q_bias_delta = per_query_masked_rms(centered_bias_delta)
    q_logits_delta = per_query_masked_rms(centered_logits_delta)
    q_tv = per_query_tv(A["weights"], D0["weights"])
    q_attended = per_query_l2(D0["attended"] - A["attended"])
    q_branch = per_query_l2(D0["branch"] - A["branch"])
    for qi in query_indices:
        if qi >= Q:
            continue
        print(f"{qi:5d} | {q_bias_delta[qi].item():23.8g} | {q_logits_delta[qi].item():24.8g} | {q_tv[qi].item():12.8g} | {q_attended[qi].item():16.8g} | {q_branch[qi].item():14.8g}")

    # End-to-end intervention: alter only the geometry metadata passed to the
    # geometry-attention module. The BEV tokens, reference, Q/K/V and content
    # attention remain untouched. This measures the final predicted-path effect.
    def shift_geometry_metadata(module, inputs):
        q_in, bev_in, ref_in, xy_in, direction_in, valid_in = inputs
        xy_new = xy_in.clone()
        xy_new[..., 1] += args.delta_m
        return (q_in, bev_in, ref_in, xy_new, direction_in, valid_in)

    shift_handle = model.geometry_attn.register_forward_pre_hook(shift_geometry_metadata)
    with torch.no_grad():
        pred_shifted = model(batch).detach().clone()
    shift_handle.remove()
    path_delta = pred_shifted - pred_baseline
    path_norm = path_delta.norm(dim=-1)
    late_start = min(60, path_delta.shape[1] - 1)
    print("\nEnd-to-end path response (geometry metadata only):")
    print("Whole-path mean displacement (m):", path_norm.mean().item())
    print("Late-path mean displacement (m):", path_norm[:, late_start:].mean().item())
    print("Final-point displacement (m):", path_norm[:, -1].mean().item())
    print("Maximum point displacement (m):", path_norm.max().item())
    print("\nDiagnostic only; no model weights were changed.")


if __name__ == "__main__":
    main()
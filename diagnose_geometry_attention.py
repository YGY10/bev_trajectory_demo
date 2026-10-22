
import torch
import torch.nn.functional as F
from datasets import BEVDataset
from tools.model_compat import load_trajectory_model

CKPT = "checkpoints/v6B1G_point_GeoAttn_5ep_best_ade.pt"
DATA = "data/v5_splits/val.jsonl"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, ckpt, info = load_trajectory_model(CKPT, device)
model.eval()

ds = BEVDataset(
    DATA, split="all", augment=False, max_lines=32
)

# Find a straight-navigation validation scene.
idx = next(
    i for i, row in enumerate(ds.rows)
    if row["navigation"] == "straight"
)
sample = ds[idx]
batch = {
    k: sample[k].unsqueeze(0).to(device)
    for k in ("lines", "line_valid", "reference", "navigation")
}

captured = {}

def capture_inputs(module, args):
    names = (
        "q", "bev", "ref", "xy", "direction", "valid"
    )
    captured.update({
        name: x.detach().clone()
        for name, x in zip(names, args)
    })

handle = model.geometry_attn.register_forward_pre_hook(
    capture_inputs
)

with torch.no_grad():
    pred = model(batch)

handle.remove()

attn = model.geometry_attn

q = captured["q"]
bev = captured["bev"]
ref = captured["ref"]
xy = captured["xy"]
direction = captured["direction"]
valid = captured["valid"]

B, Q, D = q.shape
S = bev.shape[1]
H = attn.heads
Dh = attn.head_dim

def inspect_attention(xy_input):
    tangent = torch.zeros_like(ref)
    tangent[:, 1:] = ref[:, 1:] - ref[:, :-1]
    tangent[:, 0] = tangent[:, 1]
    tangent = F.normalize(tangent, dim=-1, eps=1e-6)

    normal = torch.stack(
        [-tangent[..., 1], tangent[..., 0]], -1
    )
    delta = xy_input[:, None] - ref[:, :, None]

    ds_rel = (delta * tangent[:, :, None]).sum(-1)
    dl_rel = (delta * normal[:, :, None]).sum(-1)
    radial = torch.sqrt(
        ds_rel.square() + dl_rel.square() + 1e-6
    )

    point_dir = F.normalize(
        direction, dim=-1, eps=1e-6
    )
    align = (
        point_dir[:, None] * tangent[:, :, None]
    ).sum(-1).abs()

    geometry = torch.stack([
        (ds_rel / 10).clamp(-12, 12),
        (dl_rel / 10).clamp(-12, 12),
        (radial / 10).clamp(0, 16),
        align,
        (radial / 10).square().clamp(0, 256),
    ], -1)

    bias = attn.bias_mlp(geometry).permute(
        0, 3, 1, 2
    )

    qh = attn.q_proj(q).reshape(
        B, Q, H, Dh
    ).transpose(1, 2)

    kh = attn.k_proj(bev).reshape(
        B, S, H, Dh
    ).transpose(1, 2)

    vh = attn.v_proj(bev).reshape(
        B, S, H, Dh
    ).transpose(1, 2)

    content = (qh @ kh.transpose(-2, -1)) * (
        Dh ** -0.5
    )
    logits = content + bias
    logits = logits.masked_fill(
        ~valid[:, None, None, :],
        torch.finfo(logits.dtype).min
    )
    weights = torch.softmax(logits, -1)

    attended = (weights @ vh).transpose(
        1, 2
    ).reshape(B, Q, D)

    output = (
        torch.sigmoid(attn.gate_logit)
        * attn.out_proj(attended)
    )
    return content, bias, weights, radial, output


with torch.no_grad():
    content, bias, weights, radial, output = (
        inspect_attention(xy)
    )

    # Geometry-only intervention: change position
    # metadata but keep the encoded BEV values fixed.
    xy_shifted = xy.clone()
    xy_shifted[..., 1] += 0.5

    _, bias_shift, weights_shift, _, output_shift = (
        inspect_attention(xy_shifted)
    )

valid_flat = valid[0]
content_valid = content[0, :, :, valid_flat]
bias_valid = bias[0, :, :, valid_flat]

print("Scene:", ds.rows[idx].get("id", idx))
print("Valid BEV points:", int(valid_flat.sum()))
print("Gate:", torch.sigmoid(attn.gate_logit).item())
print("Content logits std:", content_valid.std().item())
print("Geometry bias std:", bias_valid.std().item())
print("Geometry output RMS:", output.square().mean().sqrt().item())

print("\nPer-query spatial attention:")
for qi in (20, 50, 70, 80):
    w = weights[0, :, qi, :]
    r = radial[0, qi, :]

    near5 = (
        w * ((r <= 5) & valid_flat)[None, :]
    ).sum(-1).mean().item()

    near10 = (
        w * ((r <= 10) & valid_flat)[None, :]
    ).sum(-1).mean().item()

    weighted_distance = (
        w * r[None, :]
    ).sum(-1).mean().item()

    print(
        f"Query {qi:02d}: "
        f"mass_5m={near5:.4f} "
        f"mass_10m={near10:.4f} "
        f"mean_distance={weighted_distance:.2f}m"
    )

weight_change = (
    weights_shift - weights
).abs().sum(-1).mean().item() / 2

output_change = (
    output_shift - output
).square().sum(-1).sqrt().mean().item()

print("\nGeometry-only +0.5m global-Y intervention:")
print("Mean attention TV distance:", weight_change)
print("Mean geometry output change (magnitude in feature space):",
      output_change)
print("Bias RMS change:",
      (bias_shift - bias).square().mean().sqrt().item())

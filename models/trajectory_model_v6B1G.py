"""V6B1-G: D1 point-token model + reference-centric BEV geometry attention.

Model input/output is identical to V6B1. The original global encoder/decoder and
D1 query residual remain untouched; an additional point-level geometry-conditioned
cross-attention contributes to each query before the residual head.
"""
import torch
from torch import nn
import torch.nn.functional as F
from models.trajectory_model_v6B1 import TrajectoryModel as V6B1Model


class GeometryCrossAttention(nn.Module):
    """Query-specific point attention with learned relative-geometry bias.

    The segment direction is sign-invariant because line order is randomly reversed.
    Spatial coordinates are expressed in the reference point's local tangent frame.
    """
    def __init__(self, dim=128, heads=4):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.bias_mlp = nn.Sequential(
            nn.Linear(5, 64), nn.GELU(), nn.Linear(64, heads)
        )
        # Conservative initial influence; learned during training.
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, queries, bev, ref, bev_xy, bev_dir, bev_valid):
        B, Q, D = queries.shape
        S = bev.shape[1]
        H, Dh = self.heads, self.head_dim
        if S != bev_xy.shape[1] or S != bev_dir.shape[1]:
            raise ValueError("BEV token geometry and token count mismatch")
        if not bev_valid.any(dim=1).all():
            raise ValueError("At least one real BEV polyline required per sample")

        tangent = torch.zeros_like(ref)
        tangent[:, 1:] = ref[:, 1:] - ref[:, :-1]
        tangent[:, 0] = tangent[:, 1]
        tangent = F.normalize(tangent, dim=-1, eps=1e-6)
        normal = torch.stack([-tangent[..., 1], tangent[..., 0]], -1)

        # [B,Q,S,2] with geometry in meters. Normalize displacement by
        # 10m to avoid unbounded magnitude while retaining sign and direction.
        delta = bev_xy[:, None] - ref[:, :, None]
        ds = (delta * tangent[:, :, None]).sum(-1)
        dl = (delta * normal[:, :, None]).sum(-1)
        point_dir = F.normalize(bev_dir, dim=-1, eps=1e-6)
        align = (point_dir[:, None] * tangent[:, :, None]).sum(-1).abs()
        radial = torch.sqrt(ds.square() + dl.square() + 1e-6)
        geometry = torch.stack([
            (ds / 10).clamp(-12, 12),
            (dl / 10).clamp(-12, 12),
            (radial / 10).clamp(0, 16),
            align,
            (radial / 10).square().clamp(0, 256),
        ], dim=-1)
        bias = self.bias_mlp(geometry).permute(0, 3, 1, 2)  # [B,H,Q,S]

        q = self.q_proj(queries).view(B, Q, H, Dh).transpose(1, 2)
        k = self.k_proj(bev).view(B, S, H, Dh).transpose(1, 2)
        v = self.v_proj(bev).view(B, S, H, Dh).transpose(1, 2)
        logits = (q @ k.transpose(-2, -1)) * (Dh ** -0.5) + bias
        logits = logits.masked_fill(~bev_valid[:, None, None, :],
                                    torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        attended = (weights @ v).transpose(1, 2).reshape(B, Q, D)
        return torch.sigmoid(self.gate_logit) * self.out_proj(attended)


class TrajectoryModel(V6B1Model):
    """V6B1-D1 with an additional query-local geometry branch."""
    def __init__(self, horizon=81, dim=128, heads=4, layers=2,
                 ref_points=81, bev_encoder="cnn", bev_token_mode="point"):
        if bev_token_mode != "point":
            raise ValueError("V6B1-G first experiment supports point tokens only")
        super().__init__(horizon=horizon, dim=dim, heads=heads, layers=layers,
                         ref_points=ref_points, bev_encoder=bev_encoder,
                         bev_token_mode=bev_token_mode)
        self.geometry_attn = GeometryCrossAttention(dim, heads)

    def forward(self, batch):
        lines, valid = batch["lines"], batch["line_valid"].bool()
        B, L, N, _ = lines.shape
        if self.bev_encoder == "mlp":
            pt_features = self.point_mlp(lines)
        else:
            pt_features = self.point_cnn(lines)
        bev_tokens = pt_features.reshape(B, L * N, -1)
        bev_tokens = bev_tokens + self.token_type.weight[0]
        bev_valid = valid.unsqueeze(-1).expand(-1, -1, N).reshape(B, L * N)
        bev_xy = lines[..., :2].reshape(B, L * N, 2)
        bev_dir = lines[..., 2:4].reshape(B, L * N, 2)

        ref = batch["reference"]
        if ref.shape[1] != self.horizon:
            raise ValueError("Reference count must match path horizon")
        tangents = torch.zeros_like(ref)
        tangents[:, 1:] = ref[:, 1:] - ref[:, :-1]
        tangents[:, 0] = tangents[:, 1]
        ref_tokens = self.ref_mlp(torch.cat([ref / 80.0, tangents], -1))
        ref_tokens = ref_tokens + self.token_type.weight[1]
        context = self.nav(batch["navigation"]).unsqueeze(1) + self.token_type.weight[2]
        memory = torch.cat([bev_tokens, ref_tokens, context], dim=1)
        pad = torch.cat([
            ~bev_valid,
            torch.zeros(B, ref.size(1) + 1, dtype=torch.bool, device=valid.device)
        ], dim=1)
        memory = self.encoder(memory, src_key_padding_mask=pad)
        s = torch.linspace(0., 1., self.horizon, device=ref.device, dtype=ref.dtype)
        s = s.view(1, -1, 1).expand(B, -1, -1)
        q = self.queries.unsqueeze(0) + self.query_geo(torch.cat([
            ref / 80.0, tangents, s], -1)) + context
        decoded, _ = self.decoder(q, memory, memory,
                                  key_padding_mask=pad, need_weights=False)
        geo_decoded = self.geometry_attn(
            q, memory[:, :L * N], ref, bev_xy, bev_dir, bev_valid
        )
        x = self.decoder_norm(q + decoded + geo_decoded)
        residual = self.head(x)
        pred = ref + residual
        return pred - pred[:, :1]

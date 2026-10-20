"""Time-free BEV path predictor with arc-length queries and reference correction."""
import torch
from torch import nn


class TrajectoryModel(nn.Module):
    """Historical class name kept for compatibility; the output is a PATH, not a trajectory."""
    def __init__(self, horizon=81, dim=128, heads=4, layers=2, ref_points=81):
        super().__init__()
        self.horizon = horizon
        self.point_mlp = nn.Sequential(nn.Linear(11, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.line_mlp = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.ref_mlp = nn.Sequential(nn.Linear(4, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.nav = nn.Embedding(3, dim)
        self.token_type = nn.Embedding(3, dim)
        enc_layer = nn.TransformerEncoderLayer(dim, heads, 3 * dim, 0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.queries = nn.Parameter(torch.randn(horizon, dim) * 0.02)
        # (reference x/y, local tangent x/y, path progress s/80)
        self.query_geo = nn.Sequential(nn.Linear(5, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.decoder = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 2))

    def forward(self, batch):
        lines, valid = batch['lines'], batch['line_valid']
        B = lines.size(0)
        pt_features = self.point_mlp(lines)
        line_tokens = self.line_mlp(torch.cat([
            pt_features.mean(2), pt_features.max(2).values], dim=-1)) + self.token_type.weight[0]
        ref = batch['reference']
        if ref.shape[1] != self.horizon:
            raise ValueError('Reference count must match path horizon')
        tangents = torch.zeros_like(ref)
        tangents[:, 1:] = ref[:, 1:] - ref[:, :-1]
        tangents[:, 0] = tangents[:, 1]
        ref_tokens = self.ref_mlp(torch.cat((ref / 80.0, tangents), dim=-1)) + self.token_type.weight[1]
        context = self.nav(batch['navigation']).unsqueeze(1) + self.token_type.weight[2]
        memory = torch.cat([line_tokens, ref_tokens, context], dim=1)
        pad = torch.cat([~valid, torch.zeros(B, ref.shape[1] + 1,
                        device=valid.device, dtype=torch.bool)], dim=1)
        memory = self.encoder(memory, src_key_padding_mask=pad)
        s = torch.linspace(0., 1., self.horizon, device=ref.device).view(1, -1, 1).expand(B, -1, -1)
        geo = torch.cat([ref / 80.0, tangents, s], dim=-1)
        q = self.queries.unsqueeze(0) + self.query_geo(geo) + context
        decoded, _ = self.decoder(q, memory, memory, key_padding_mask=pad, need_weights=False)
        residual = self.head(decoded)
        # Output is an 80 m geometric path with no time or velocity coordinates.
        pred = ref + residual
        return pred - pred[:, :1]  # exact ego-frame start at (0,0)

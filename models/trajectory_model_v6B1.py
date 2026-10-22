
"""Time-free BEV path predictor with selectable per-Polyline point encoder."""
import torch
from torch import nn
import torch.nn.functional as F


class ResidualConv1dBlock(nn.Module):
    """1D convolution residual block for ordered polyline points."""

    def __init__(self, dim):
        super().__init__()

        self.conv1 = nn.Conv1d(
            dim, dim, kernel_size=3, padding=1
        )
        self.norm1 = nn.LayerNorm(dim)

        self.conv2 = nn.Conv1d(
            dim, dim, kernel_size=3, padding=1
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # Input: [B*L, D, N]
        residual = x

        # Conv1d: [B*L, D, N]
        y = self.conv1(x)

        # LayerNorm: [B*L, N, D]
        y = y.transpose(1, 2)
        y = self.norm1(y)
        y = F.gelu(y)

        # Back to Conv1d format
        y = y.transpose(1, 2)

        y = self.conv2(y)

        # LayerNorm over feature dimension
        y = y.transpose(1, 2)
        y = self.norm2(y)

        # Back to Conv1d format
        y = y.transpose(1, 2)

        return F.gelu(residual + y)


class PolylineCNNEncoder(nn.Module):
    """Encode each ordered polyline independently.

    Input:  [B, L, N, 11]
    Output: [B, L, N, D]
    """

    def __init__(self, in_features=11, dim=128):
        super().__init__()
        self.stem = nn.Conv1d(
            in_features, dim, kernel_size=3, padding=1
        )
        self.stem_norm = nn.LayerNorm(dim)
        self.block = ResidualConv1dBlock(dim)

    def forward(self, lines):
        batch_size, num_lines, num_points, num_features = lines.shape

        # Treat every polyline as an independent ordered sequence.
        x = lines.reshape(
            batch_size * num_lines, num_points, num_features
        ).transpose(1, 2)

        x = self.stem(x).transpose(1, 2)
        x = F.gelu(self.stem_norm(x)).transpose(1, 2)
        x = self.block(x)

        return x.transpose(1, 2).reshape(
            batch_size, num_lines, num_points, -1
        )


class TrajectoryModel(nn.Module):
    """Predict a geometric path, not a time-parameterized trajectory."""

    def __init__(
        self,
        horizon=81,
        dim=128,
        heads=4,
        layers=2,
        ref_points=81,
        bev_encoder="mlp",
        bev_token_mode="pooled",
    ):
        super().__init__()
        self.horizon = horizon
        self.bev_encoder = bev_encoder
        self.bev_token_mode = bev_token_mode;
        self.decoder_norm = nn.LayerNorm(dim)

        if bev_token_mode not in ("pooled", "point"):
            raise ValueError(
                f"Unknown BEV token mode: {bev_token_mode}"
            )

        if bev_encoder == "mlp":
            # Original V5 encoder: keep its parameter names unchanged
            # so existing MLP checkpoints remain compatible.
            self.point_mlp = nn.Sequential(
                nn.Linear(11, dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
            )
        elif bev_encoder == "cnn":
            self.point_cnn = PolylineCNNEncoder(
                in_features=11, dim=dim
            )
        else:
            raise ValueError(
                f"Unknown BEV encoder: {bev_encoder!r}"
            )

        if self.bev_token_mode == "pooled":
            self.line_mlp = nn.Sequential(
                nn.Linear(2 * dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
            )
        self.ref_mlp = nn.Sequential(
            nn.Linear(4, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.nav = nn.Embedding(3, dim)
        self.token_type = nn.Embedding(3, dim)

        enc_layer = nn.TransformerEncoderLayer(
            dim, heads, 3 * dim, 0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, layers)

        self.queries = nn.Parameter(
            torch.randn(horizon, dim) * 0.02
        )
        self.query_geo = nn.Sequential(
            nn.Linear(5, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.decoder = nn.MultiheadAttention(
            dim, heads, batch_first=True
        )
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 2),
        )

    def forward(self, batch):
        lines, valid = batch["lines"], batch["line_valid"]
        B = lines.size(0)

        if self.bev_encoder == "mlp":
            pt_features = self.point_mlp(lines)
        else:
            pt_features = self.point_cnn(lines)
        
        # BEV tokenization:
        # pooled: one token per polyline (original V6B1)
        # point:  one token per observed point (new experiment)

        if self.bev_token_mode == "pooled":
            bev_tokens = self.line_mlp(torch.cat([
                pt_features.mean(dim=2),
                pt_features.max(dim=2).values,
            ], dim=-1))
            bev_tokens = bev_tokens + self.token_type.weight[0]

            # [B, L]
            bev_valid = valid

        else:
            # pt_features: [B, L, N, D]
            B, L, N, D = pt_features.shape

            # Keep every point feature. No mean/max pooling.
            # [B, L, N, D] -> [B, L*N, D]
            bev_tokens = pt_features.reshape(B, L * N, D)
            bev_tokens = bev_tokens + self.token_type.weight[0]

            # A padded polyline makes all its point tokens invalid.
            # [B, L] -> [B, L*N]
            bev_valid = (
                valid.unsqueeze(-1)
                .expand(-1, -1, N)
                .reshape(B, L * N)
            )


        ref = batch["reference"]
        if ref.shape[1] != self.horizon:
            raise ValueError(
                "Reference count must match path horizon"
            )

        tangents = torch.zeros_like(ref)
        tangents[:, 1:] = ref[:, 1:] - ref[:, :-1]
        tangents[:, 0] = tangents[:, 1]

        ref_tokens = self.ref_mlp(
            torch.cat((ref / 80.0, tangents), dim=-1)
        ) + self.token_type.weight[1]

        context = (
            self.nav(batch["navigation"]).unsqueeze(1)
            + self.token_type.weight[2]
        )

        memory = torch.cat(
            [bev_tokens, ref_tokens, context], dim=1
        )

        pad = torch.cat([
            ~bev_valid,
            torch.zeros(
                B,
                ref.shape[1] + 1,
                device=valid.device,
                dtype=torch.bool,
            ),
        ], dim=1)

        memory = self.encoder(
            memory, src_key_padding_mask=pad
        )

        s = torch.linspace(
            0., 1., self.horizon, device=ref.device
        ).view(1, -1, 1).expand(B, -1, -1)

        geo = torch.cat([ref / 80.0, tangents, s], dim=-1)
        q = self.queries.unsqueeze(0) + self.query_geo(geo) + context

        decoded, _ = self.decoder(
            q, memory, memory,
            key_padding_mask=pad,
            need_weights=False,
        )

        x = self.decoder_norm(q + decoded)
        residual = self.head(x)
        pred = ref + residual
        return pred - pred[:, :1]

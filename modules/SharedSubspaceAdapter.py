import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class SSA(nn.Module):
    def __init__(
        self,
        in_dim,
        sar_dim=None,
        factor=4,
        hidden_dim=None,
        shared_ratio=0.5,
        drop=0.1,
    ):
        super().__init__()
        sar_dim = sar_dim or in_dim
        hidden_dim = hidden_dim or min(96, max(24, in_dim // factor))

        shared_dim = max(8, int(round(hidden_dim * shared_ratio)))
        shared_dim = min(shared_dim, hidden_dim - 8)
        private_dim = hidden_dim - shared_dim
        if private_dim <= 0:
            raise ValueError("hidden_dim is too small for private/shared split.")

        self.private_dim = private_dim

        self.rgb_norm = nn.LayerNorm(in_dim)
        self.sar_norm = nn.LayerNorm(sar_dim)
        self.rgb_proj = nn.Linear(in_dim, hidden_dim)
        self.sar_proj = nn.Linear(sar_dim, hidden_dim)

        self.rgb_private_refine = nn.Linear(private_dim, private_dim)
        self.sar_private_refine = nn.Linear(private_dim, private_dim)
        self.shared_refine = nn.Linear(shared_dim, shared_dim, bias=False)

        self.dropout = nn.Dropout(p=drop)
        self.rgb_out = nn.Linear(hidden_dim, in_dim)
        self.sar_out = nn.Linear(hidden_dim, sar_dim)

        self.exchange_alpha = nn.Parameter(torch.tensor(1.0))
        self.exchange_beta = nn.Parameter(torch.tensor(0.0))
        self.rgb_scale = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.sar_scale = nn.Parameter(torch.ones(sar_dim) * 1e-6)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _split_subspaces(self, x):
        return x[..., :self.private_dim], x[..., self.private_dim:]

    def forward(self, x_rgb, x_sar):
        rgb_hidden = self.rgb_proj(self.rgb_norm(x_rgb))
        sar_hidden = self.sar_proj(self.sar_norm(x_sar))

        rgb_private, rgb_shared = self._split_subspaces(rgb_hidden)
        sar_private, sar_shared = self._split_subspaces(sar_hidden)

        rgb_private = F.gelu(self.rgb_private_refine(rgb_private))
        sar_private = F.gelu(self.sar_private_refine(sar_private))

        shared_delta = self.shared_refine(sar_shared - rgb_shared)
        exchange_score = shared_delta.abs().mean(dim=-1, keepdim=True)
        shared_shift = torch.sigmoid(
            self.exchange_alpha * exchange_score + self.exchange_beta) * shared_delta

        rgb_fused = torch.cat([rgb_private, rgb_shared + shared_shift], dim=-1)
        sar_fused = torch.cat([sar_private, sar_shared - shared_shift], dim=-1)

        rgb_delta = self.rgb_out(self.dropout(F.gelu(rgb_fused))) * self.rgb_scale
        sar_delta = self.sar_out(self.dropout(F.gelu(sar_fused))) * self.sar_scale

        return x_rgb + rgb_delta, x_sar + sar_delta

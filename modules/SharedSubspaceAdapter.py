"""Shared Subspace Adapter (SSA).

Block-level adapter inserted after the FFN of each frozen Swin-Large block.
Decomposes features into private and shared subspaces, then performs
anti-symmetric exchange via a gated shared-delta shift.  One of the three
AXIA fusion modules alongside DGA (block-level) and LSGF (stage-level).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from timm.layers import trunc_normal_
except ImportError:  # older timm
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

        self.hidden_dim = hidden_dim
        self.private_dim = private_dim
        self.shared_dim = shared_dim

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
        private = x[..., :self.private_dim]
        shared = x[..., self.private_dim:]
        return private, shared

    def forward(self, x_rgb, x_sar, hw_shapes=None):
        del hw_shapes  # The exchange is token-wise and does not require spatial reshaping.

        rgb_identity = x_rgb
        sar_identity = x_sar

        rgb_hidden = self.rgb_proj(self.rgb_norm(x_rgb))
        sar_hidden = self.sar_proj(self.sar_norm(x_sar))

        rgb_private, rgb_shared = self._split_subspaces(rgb_hidden)
        sar_private, sar_shared = self._split_subspaces(sar_hidden)

        rgb_private = F.gelu(self.rgb_private_refine(rgb_private))
        sar_private = F.gelu(self.sar_private_refine(sar_private))

        shared_delta = self.shared_refine(sar_shared - rgb_shared)
        exchange_score = shared_delta.abs().mean(dim=-1, keepdim=True)
        exchange_gate = torch.sigmoid(self.exchange_alpha * exchange_score + self.exchange_beta)
        shared_shift = exchange_gate * shared_delta

        rgb_shared = rgb_shared + shared_shift
        sar_shared = sar_shared - shared_shift

        rgb_fused = torch.cat([rgb_private, rgb_shared], dim=-1)
        sar_fused = torch.cat([sar_private, sar_shared], dim=-1)

        rgb_delta = self.rgb_out(self.dropout(F.gelu(rgb_fused))) * self.rgb_scale
        sar_delta = self.sar_out(self.dropout(F.gelu(sar_fused))) * self.sar_scale

        return rgb_identity + rgb_delta, sar_identity + sar_delta

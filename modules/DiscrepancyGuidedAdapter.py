"""Discrepancy-Guided Adapter (DGA).

Block-level adapter inserted after the window attention of each frozen
Swin-Large block.  Uses cross-modal cosine discrepancy as a spatial routing
signal to selectively adapt the RGB feature stream.  One of the three AXIA
fusion modules alongside SSA (block-level) and LSGF (stage-level).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from timm.layers import trunc_normal_
except ImportError:  # older timm
    from timm.models.layers import trunc_normal_


def _spatial_norm(num_channels):
    num_groups = min(8, num_channels)
    while num_groups > 1 and num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class _LiteMixer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = _spatial_norm(channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = x + self.dw(x)
        x = self.pw(x)
        return self.act(self.norm(x))


class DGA(nn.Module):

    def __init__(
        self,
        in_dim,
        sar_dim=None,
        factor=4,
        hidden_dim=None,
        drop=0.1,
        rgb_drop_prob=0.15,
        route_kernel_size=3,
    ):
        super().__init__()
        sar_dim = sar_dim or in_dim
        hidden_dim = hidden_dim or min(96, max(24, in_dim // factor))
        padding = route_kernel_size // 2

        self.rgb_norm = nn.LayerNorm(in_dim)
        self.sar_norm = nn.LayerNorm(sar_dim)
        self.rgb_proj = nn.Linear(in_dim, hidden_dim)
        self.sar_proj = nn.Linear(sar_dim, hidden_dim)

        self.rgb_mixer = _LiteMixer(hidden_dim)
        self.rgb_dropout = nn.Dropout2d(p=rgb_drop_prob)
        self.route_smoother = nn.Conv2d(1, 1, kernel_size=route_kernel_size, padding=padding, bias=True)

        self.dropout = nn.Dropout(p=drop)
        self.out_proj = nn.Linear(hidden_dim, in_dim)

        self.route_alpha = nn.Parameter(torch.tensor(1.0))
        self.route_beta = nn.Parameter(torch.tensor(0.0))
        self.out_scale = nn.Parameter(torch.ones(in_dim) * 1e-6)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            if getattr(m, "weight", None) is not None:
                nn.init.constant_(m.weight, 1.0)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _tokens_to_map(x, hw_shapes):
        b, n, c = x.shape
        h, w = hw_shapes
        if n != h * w:
            raise ValueError(f"Token count {n} does not match spatial size {h}x{w}.")
        return x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _map_to_tokens(x):
        b, c, h, w = x.shape
        return x.permute(0, 2, 3, 1).reshape(b, h * w, c).contiguous()

    def forward(self, x_rgb, x_sar, hw_shapes=None):
        if hw_shapes is None:
            raise ValueError("hw_shapes must be provided for DGA.")

        identity = x_rgb

        rgb_tokens = self.rgb_proj(self.rgb_norm(x_rgb))
        sar_tokens = self.sar_proj(self.sar_norm(x_sar))

        rgb_map = self._tokens_to_map(rgb_tokens, hw_shapes)
        sar_map = self._tokens_to_map(sar_tokens, hw_shapes)

        rgb_update = self.rgb_mixer(rgb_map)       # depthwise-separable conv
        rgb_update = self.rgb_dropout(rgb_update)  # drop RGB updates to avoid over-reliance

        rgb_unit = F.normalize(rgb_map, dim=1, eps=1e-6)
        sar_unit = F.normalize(sar_map, dim=1, eps=1e-6)
        conflict = 1.0 - F.cosine_similarity(rgb_unit, sar_unit, dim=1, eps=1e-6).unsqueeze(1)

        route = self.route_smoother(conflict)
        gate = torch.sigmoid(self.route_alpha * route + self.route_beta)
        fused = rgb_update * gate

        fused_tokens = self._map_to_tokens(fused)
        fused_tokens = self.dropout(F.gelu(fused_tokens))
        delta = self.out_proj(fused_tokens) * self.out_scale

        return identity + delta

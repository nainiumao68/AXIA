import torch
import torch.nn as nn
import torch.nn.functional as F
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
        temperature=1.0,
    ):
        super().__init__()
        if float(temperature) <= 0:
            raise ValueError(f"DGA temperature must be > 0, got {temperature}.")
        self.temperature = float(temperature)

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
        rgb_tokens = self.rgb_proj(self.rgb_norm(x_rgb))
        rgb_map = self._tokens_to_map(rgb_tokens, hw_shapes)
        rgb_update = self.rgb_dropout(self.rgb_mixer(rgb_map))

        sar_tokens = self.sar_proj(self.sar_norm(x_sar))
        sar_map = self._tokens_to_map(sar_tokens, hw_shapes)
        conflict = 1.0 - F.cosine_similarity(
            F.normalize(rgb_map, dim=1, eps=1e-6),
            F.normalize(sar_map, dim=1, eps=1e-6),
            dim=1, eps=1e-6).unsqueeze(1)
        gate = torch.sigmoid(self.temperature * (
            self.route_alpha * self.route_smoother(conflict) + self.route_beta))

        fused_tokens = self._map_to_tokens(rgb_update * gate)
        delta = self.out_proj(self.dropout(F.gelu(fused_tokens))) * self.out_scale

        return x_rgb + delta

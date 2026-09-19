"""Scene-Aware Reassembly Decoder (SARD).

Top-down FPN decoder with scene-conditioned dual-axis context
(SceneGuidedChannelReassembly) and cascaded multi-kernel spatial fusion
(CascadedMultiKernelPerception), plus full multi-scale tail aggregation.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _dual_pool_descriptor(x):
    """GAP / GMP parallel -> compact global channel descriptor."""
    avg_desc = F.adaptive_avg_pool2d(x, 1).flatten(1)
    max_desc = F.adaptive_max_pool2d(x, 1).flatten(1)
    return torch.cat([avg_desc, max_desc], dim=1)


def _resolve_groups(channels, max_groups=8):
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


class DepthwiseKernelBranch(nn.Module):

    def __init__(self, channels, kernel_size, dilation=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            norm_layer(channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class SceneGuidedChannelReassembly(nn.Module):
    """Spatial-Channel dual-axis context module for the deepest stage.

    Design:
      1. Spatial axis: large-kernel depthwise conv encodes local/mid-range
         spatial structure at ~1/32 resolution (covers nearly half the map).
      2. Channel axis: GAP/GMP scene descriptor drives a low-rank group
         mixing matrix for cross-group semantic transfer.
      3. The two axes are orthogonal: spatial conv handles position-aware
         structure, group mixing handles channel-wise semantic reorganization.
    """

    def __init__(self, channels, norm_layer=nn.BatchNorm2d, reduction=4,
                 max_groups=8, mix_rank=4):
        super().__init__()
        hidden = max(channels // reduction, 64)
        self.groups = _resolve_groups(channels, max_groups=max_groups)
        self.group_channels = channels // self.groups
        self.mix_rank = max(1, min(int(mix_rank), self.groups))

        self.spatial_inject = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                      groups=channels, bias=False),
            norm_layer(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            norm_layer(channels),
        )

        self.scene_proj = nn.Sequential(
            nn.Linear(channels * 2, hidden, bias=False),
            nn.GELU(),
        )
        self.group_left = nn.Linear(hidden, self.groups * self.mix_rank, bias=False)
        self.group_right = nn.Linear(hidden, self.groups * self.mix_rank, bias=False)
        self.identity_bias = nn.Parameter(torch.tensor(1.0))
        self.register_buffer(
            "group_identity",
            torch.eye(self.groups, dtype=torch.float32).unsqueeze(0),
            persistent=False,
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            norm_layer(channels),
            nn.GELU(),
        )

    def forward(self, x):
        b, c, h, w = x.shape

        # 1) Spatial branch via depthwise convolution.
        selected = x + self.spatial_inject(x)

        # 2) Global scene descriptor driving the conditional group-mixing matrix.
        descriptor = _dual_pool_descriptor(selected)    # (b, 2c)
        scene_token = self.scene_proj(descriptor)       # (b, hidden)

        # 3) Low-rank group mixing: cross-group semantic transfer.
        left = self.group_left(scene_token).view(b, self.groups, self.mix_rank)
        right = self.group_right(scene_token).view(b, self.mix_rank, self.groups)
        mix_logits = torch.bmm(left, right) / math.sqrt(float(self.mix_rank))
        mix_logits = mix_logits + self.identity_bias * self.group_identity
        mix_weights = torch.softmax(mix_logits, dim=-1)    # (b, groups, groups)

        grouped = selected.view(b, self.groups, self.group_channels, h, w)
        mixed = torch.einsum("bij,bjchw->bichw", mix_weights, grouped)
        mixed = mixed.reshape(b, c, h, w)

        # 4) Residual write-back.
        refined = self.out_proj(mixed)
        return x + refined


class CascadedMultiKernelPerception(nn.Module):
    """Dual-pool guided multi-kernel cascade merge.

    GAP/GMP extracts channel priors, then routes across multiple conv bases:
    3x3 depthwise for local texture, 7x7 for mid-range structure,
    dilated 3x3 for sparse context. The shallow feature serves as the
    reconstruction anchor.
    """

    def __init__(self, channels, norm_layer=nn.BatchNorm2d, align_corners=False,
                 reduction=4):
        super().__init__()
        self.align_corners = align_corners
        hidden = max(channels // reduction, 64)

        self.fusion_proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            norm_layer(channels),
            nn.GELU(),
        )

        self.prior_proj = nn.Sequential(
            nn.Linear(channels * 2, hidden, bias=False),
            nn.GELU(),
        )
        self.channel_gate = nn.Linear(hidden, channels, bias=False)
        self.branch_router = nn.Linear(hidden, 3, bias=True)

        self.branches = nn.ModuleList([
            DepthwiseKernelBranch(channels, kernel_size=3, dilation=1, norm_layer=norm_layer),
            DepthwiseKernelBranch(channels, kernel_size=7, dilation=1, norm_layer=norm_layer),
            DepthwiseKernelBranch(channels, kernel_size=3, dilation=2, norm_layer=norm_layer),
        ])

        self.mix_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            norm_layer(channels),
            nn.GELU(),
        )
        self.reconstruct = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            norm_layer(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            norm_layer(channels),
            nn.GELU(),
        )

    def forward(self, deep, shallow):
        deep_up = F.interpolate(
            deep,
            size=shallow.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners,
        )
        fusion = self.fusion_proj(torch.cat([deep_up, shallow], dim=1))

        descriptor = _dual_pool_descriptor(fusion)
        prior = self.prior_proj(descriptor)
        channel_gate = torch.sigmoid(self.channel_gate(prior)).view(
            fusion.shape[0], fusion.shape[1], 1, 1)
        branch_weights = torch.softmax(self.branch_router(prior), dim=-1)
        branch_weights = branch_weights.view(fusion.shape[0], len(self.branches), 1, 1, 1)

        fusion = fusion * (1.0 + channel_gate)
        branch_outputs = torch.stack([branch(fusion) for branch in self.branches], dim=1)

        mixed = (branch_outputs * branch_weights).sum(dim=1)
        mixed = self.mix_refine(mixed)

        recon = self.reconstruct(torch.cat([mixed, shallow], dim=1))
        return shallow + recon


class SARD(nn.Module):
    """Scene-Aware Reassembly Decoder.

    Flow:
      1. Lateral projection to a uniform channel dim.
      2. Deepest stage: SceneGuidedChannelReassembly (spatial + channel).
      3. Top-down cascade: CascadedMultiKernelPerception at each level.
      4. Multi-scale tail: all cascade outputs upsampled, concatenated, fused.
      5. Final refine + segmentation head.
    """

    def __init__(
        self,
        in_channels=(96, 192, 384, 768),
        num_classes=40,
        channels=512,
        pool_scales=(1, 2, 3, 6),
        norm_layer=nn.BatchNorm2d,
        dropout_ratio=0.1,
        align_corners=False,
        **kwargs,
    ):
        super().__init__()
        del pool_scales
        self.channels = channels
        self.align_corners = align_corners
        n_levels = len(in_channels)

        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(inc, channels, kernel_size=1, bias=False),
                norm_layer(channels),
                nn.GELU(),
            )
            for inc in in_channels
        ])

        self.deep_context = SceneGuidedChannelReassembly(channels, norm_layer)

        self.merges = nn.ModuleList([
            CascadedMultiKernelPerception(channels, norm_layer, align_corners)
            for _ in range(n_levels - 1)
        ])

        self.ms_fusion = nn.Sequential(
            nn.Conv2d(channels * n_levels, channels, kernel_size=1, bias=False),
            norm_layer(channels),
            nn.GELU(),
        )

        self.final_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            norm_layer(channels),
            nn.GELU(),
        )

        self.dropout = nn.Dropout2d(p=dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.segmentation_head = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, inputs):
        feats = [lat(x) for lat, x in zip(self.laterals, inputs)]

        feats[-1] = self.deep_context(feats[-1])

        x = feats[-1]
        ms_outputs = [feats[-1]]
        for i, merge in enumerate(self.merges):
            x = merge(x, feats[len(feats) - 2 - i])
            ms_outputs.append(x)

        target_size = x.shape[2:]
        aligned = []
        for feat in ms_outputs:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear',
                                     align_corners=self.align_corners)
            aligned.append(feat)
        x = self.ms_fusion(torch.cat(aligned, dim=1))

        x = self.final_refine(x)
        x = self.dropout(x)
        return self.segmentation_head(x)


__all__ = ["SARD", "SceneGuidedChannelReassembly", "CascadedMultiKernelPerception"]

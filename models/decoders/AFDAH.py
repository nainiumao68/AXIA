"""Adaptive Frequency Disentangled Auxiliary Head (AFDAH).

Training-only auxiliary decoder operating in the frequency domain.

Design:
  (A) Scene-Adaptive Frequency Decomposer (SAFD):
      Per-stage, per-image adaptive frequency decomposition.  A tiny MLP
      predicts the interpolation ratio between a small-kernel and
      large-kernel average-pooling, making the low/high frequency split
      point scene-conditioned rather than fixed.

  (B) Intra-Band Cross-Scale Attention (IBCSA):
      Within each frequency band, self-attention across 4 scale tokens
      per spatial position decides which encoder stage contributes the
      most relevant information.  High-freq tokens attend only to
      high-freq tokens from other scales (and likewise for low-freq),
      preventing inter-band contamination.

  (C) Frequency-Aware Class Router (FACR):
      A lightweight per-pixel network predicts routing weights [w_high,
      w_low] that blend the two frequency-band representations.
      Boundary / small-object pixels are expected to route toward
      high-freq; large uniform regions toward low-freq.

  (D) Dual-Band Independent Supervision + Frequency Consistency:
      Three classification heads (fused, high-band, low-band) with
      independent CE losses force each band to be discriminative on its
      own.  An MSE consistency term between the two band predictions
      regularises agreement without conflicting gradients.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# (A) Scene-Adaptive Frequency Decomposer
# ---------------------------------------------------------------------------

class SceneAdaptiveFreqDecomposer(nn.Module):
    """Per-stage scene-adaptive frequency decomposition.

    Two fixed average-pooling kernels (k_small, k_large) are applied with
    stride=1, producing two spatial smoothings at different scales.
    A scene descriptor (GAP) predicts an interpolation ratio between them,
    yielding a *continuous* low-frequency component whose bandwidth adapts
    to the current image content.  The high-frequency residual is simply
    ``x - low``.
    """

    def __init__(self, in_channels, k_small=3, k_large=7):
        super().__init__()
        self.k_small = k_small
        self.k_large = k_large
        self.pad_small = k_small // 2
        self.pad_large = k_large // 2
        hidden = max(in_channels // 8, 16)
        self.ratio_pred = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B = x.shape[0]
        desc = F.adaptive_avg_pool2d(x, 1).flatten(1)
        ratio = self.ratio_pred(desc).view(B, 1, 1, 1)

        low_s = F.avg_pool2d(x, self.k_small, stride=1, padding=self.pad_small)
        low_l = F.avg_pool2d(x, self.k_large, stride=1, padding=self.pad_large)

        low = ratio * low_l + (1.0 - ratio) * low_s
        high = x - low
        return low, high


# ---------------------------------------------------------------------------
# (B) Intra-Band Cross-Scale Attention
# ---------------------------------------------------------------------------

class IntraBandCrossScaleAttention(nn.Module):
    """Self-attention across scale tokens *within* a single frequency band.

    At each spatial position the features from different encoder stages
    form a set of *scale tokens*.  Multi-head self-attention determines the
    optimal information mix, and a 1x1 conv aggregates the attention-refined
    S*d channels back to d.
    """

    def __init__(self, d, num_scales=4, num_heads=4):
        super().__init__()
        self.d = d
        self.num_heads = num_heads
        self.head_dim = d // num_heads

        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

        self.scale_embed = nn.Parameter(torch.randn(num_scales, d) * 0.02)
        self.agg = nn.Sequential(
            nn.Conv2d(d * num_scales, d, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.GELU(),
        )

    def forward(self, scale_feats):
        """Args: list of S tensors, each (B, d, H, W) at the same resolution."""
        B, d, H, W = scale_feats[0].shape
        S = len(scale_feats)

        x = torch.stack(scale_feats, dim=1)                      # (B, S, d, H, W)
        x = x + self.scale_embed[None, :S, :, None, None]
        x = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, S, d)   # (BHW, S, d)

        nh, hd = self.num_heads, self.head_dim
        Q = self.q(x).view(-1, S, nh, hd).transpose(1, 2)
        K = self.k(x).view(-1, S, nh, hd).transpose(1, 2)
        V = self.v(x).view(-1, S, nh, hd).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * (hd ** -0.5)
        out = (attn.softmax(dim=-1) @ V).transpose(1, 2).reshape(-1, S, d)
        out = self.out(out)

        out = out.view(B, H, W, S, d).permute(0, 3, 4, 1, 2)    # (B, S, d, H, W)
        return self.agg(out.reshape(B, S * d, H, W))              # (B, d, H, W)


# ---------------------------------------------------------------------------
# (C) Frequency-Aware Class Router
# ---------------------------------------------------------------------------

class FreqAwareClassRouter(nn.Module):
    """Per-pixel learned routing between high-freq and low-freq features."""

    def __init__(self, d):
        super().__init__()
        self.router = nn.Sequential(
            nn.Conv2d(d * 2, d, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.GELU(),
            nn.Conv2d(d, d, 3, padding=1, groups=d, bias=False),
            nn.BatchNorm2d(d),
            nn.GELU(),
            nn.Conv2d(d, 2, 1, bias=False),
        )

    def forward(self, high_feat, low_feat):
        w = self.router(torch.cat([high_feat, low_feat], dim=1)).softmax(dim=1)
        return w[:, 0:1] * high_feat + w[:, 1:2] * low_feat


# ---------------------------------------------------------------------------
# (D) Full auxiliary head
# ---------------------------------------------------------------------------

class AFDAH(nn.Module):
    """Adaptive Frequency Disentangled Auxiliary Head.

    Disabled at inference (training-only).

    Args:
        in_channels:    per-stage encoder channel dims.
        num_classes:    segmentation classes.
        embed_dim:      internal feature dimension (default 128).
        num_heads:      attention heads in cross-scale attention.
        aux_weight:     loss weight for the fused prediction.
        band_weight:    loss weight for each frequency-band prediction.
        consist_weight: loss weight for frequency consistency MSE.
    """

    def __init__(self, in_channels=(96, 192, 384, 768), num_classes=40,
                 embed_dim=128, num_heads=4,
                 aux_weight=0.4, band_weight=0.15, consist_weight=0.05):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.aux_weight = aux_weight
        self.band_weight = band_weight
        self.consist_weight = consist_weight
        n = len(in_channels)

        # (A) per-stage decomposers
        self.decomposers = nn.ModuleList([
            SceneAdaptiveFreqDecomposer(c) for c in in_channels
        ])

        # per-stage, per-band projection to embed_dim
        self.high_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embed_dim, 1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
            ) for c in in_channels
        ])
        self.low_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embed_dim, 1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
            ) for c in in_channels
        ])

        # (B) intra-band cross-scale attention
        self.high_attn = IntraBandCrossScaleAttention(embed_dim, n, num_heads)
        self.low_attn = IntraBandCrossScaleAttention(embed_dim, n, num_heads)

        # (C) frequency router
        self.router = FreqAwareClassRouter(embed_dim)

        # (D) three classification heads
        self.head_main = nn.Conv2d(embed_dim, num_classes, 1)
        self.head_high = nn.Conv2d(embed_dim, num_classes, 1)
        self.head_low = nn.Conv2d(embed_dim, num_classes, 1)

    # ---- forward ------------------------------------------------------------

    def forward(self, features, target_size):
        """
        Args:
            features:    list of 4 encoder stage tensors.
            target_size: (H, W) for final up-sampling.
        Returns:
            logits_main, logits_high, logits_low  – each (B, C, H, W).
        """
        highs, lows = [], []
        for i, (feat, dec) in enumerate(zip(features, self.decomposers)):
            lo, hi = dec(feat)
            highs.append(self.high_projs[i](hi))
            lows.append(self.low_projs[i](lo))

        # high-freq band at stage-1 resolution, low-freq at stage-2
        h_size = features[1].shape[2:]
        l_size = features[2].shape[2:]

        h_aligned = [
            F.interpolate(h, h_size, mode='bilinear', align_corners=False)
            if h.shape[2:] != h_size else h
            for h in highs
        ]
        l_aligned = [
            F.interpolate(l, l_size, mode='bilinear', align_corners=False)
            if l.shape[2:] != l_size else l
            for l in lows
        ]

        H = self.high_attn(h_aligned)
        L = self.low_attn(l_aligned)
        L_up = F.interpolate(L, H.shape[2:], mode='bilinear', align_corners=False)
        fused = self.router(H, L_up)

        def _up(t):
            return F.interpolate(t, target_size, mode='bilinear', align_corners=False)

        return _up(self.head_main(fused)), _up(self.head_high(H)), _up(self.head_low(L_up))

    # ---- loss ----------------------------------------------------------------

    def compute_loss(self, features, label, target_size, criterion):
        """Combined auxiliary loss (call this from the training loop).

        Returns a single scalar:
            aux_weight * CE_fused
          + band_weight * (CE_high + CE_low)
          + consist_weight * MSE(softmax(high), softmax(low))
        """
        logits_main, logits_high, logits_low = self.forward(features, target_size)
        gt = label.long()

        loss_main = criterion(logits_main, gt)
        loss_high = criterion(logits_high, gt)
        loss_low = criterion(logits_low, gt)

        loss_consist = F.mse_loss(
            logits_high.softmax(dim=1),
            logits_low.softmax(dim=1),
        )

        return (self.aux_weight * loss_main
                + self.band_weight * (loss_high + loss_low)
                + self.consist_weight * loss_consist)


__all__ = ["AFDAH"]

"""SegFormer-style MLP decoder.

Consumes the four AXIA encoder stages: each scale is projected by a linear
embedding, upsampled to the finest resolution, concatenated, fused, and
classified.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Linear embedding of a spatial feature map."""

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class MLPDecoder(nn.Module):
    def __init__(self, in_channels=(96, 192, 384, 768), num_classes=40,
                 dropout_ratio=0.1, norm_layer=nn.BatchNorm2d,
                 channels=256, align_corners=False, **kwargs):
        super(MLPDecoder, self).__init__()
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.in_channels = list(in_channels)
        embedding_dim = channels

        self.linear_c = nn.ModuleList([
            MLP(input_dim=c, embed_dim=embedding_dim) for c in self.in_channels
        ])

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * len(self.in_channels), embedding_dim, kernel_size=1),
            norm_layer(embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        c1 = inputs[0]
        n = c1.shape[0]
        fused = []
        for feat, mlp in zip(inputs, self.linear_c):
            _c = mlp(feat).permute(0, 2, 1).reshape(n, -1, feat.shape[2], feat.shape[3])
            _c = F.interpolate(_c, size=c1.shape[2:], mode='bilinear',
                               align_corners=self.align_corners)
            fused.append(_c)
        x = self.linear_fuse(torch.cat(fused, dim=1))
        x = self.dropout(x)
        return self.linear_pred(x)


__all__ = ["MLPDecoder"]

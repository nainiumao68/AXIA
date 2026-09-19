"""FCN decoder over four encoder stages.

AXIA's encoder emits four stages, so we 1x1-project all of them, upsample to
the finest scale, concatenate, and then apply the FCN conv stack.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNHead(nn.Module):
    def __init__(self, in_channels=(96, 192, 384, 768), num_classes=40,
                 channels=256, dropout_ratio=0.1, norm_layer=nn.BatchNorm2d,
                 **kwargs):
        super(FCNHead, self).__init__()
        self.align_corners = False
        n_levels = len(in_channels)

        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(inc, channels, kernel_size=1, bias=False),
                norm_layer(channels),
                nn.ReLU(inplace=True),
            )
            for inc in in_channels
        ])

        self.fcn = nn.Sequential(
            nn.Conv2d(channels * n_levels, channels, kernel_size=3, padding=1, bias=False),
            norm_layer(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_ratio) if dropout_ratio > 0 else nn.Identity(),
            nn.Conv2d(channels, num_classes, kernel_size=1),
        )

    def forward(self, inputs):
        feats = [lat(x) for lat, x in zip(self.laterals, inputs)]
        target_size = feats[0].shape[2:]
        aligned = [
            F.interpolate(f, size=target_size, mode='bilinear',
                          align_corners=self.align_corners)
            if f.shape[2:] != target_size else f
            for f in feats
        ]
        return self.fcn(torch.cat(aligned, dim=1))


__all__ = ["FCNHead"]

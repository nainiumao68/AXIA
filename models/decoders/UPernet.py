"""UPerNet decoder (Unified Perceptual Parsing).

Consumes the four AXIA encoder stages.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet."""

    def __init__(self, pool_scales, in_channels, channels, norm_layer, align_corners=False):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    nn.Conv2d(self.in_channels, self.channels, 1, bias=False),
                    norm_layer(self.channels),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, x):
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = F.interpolate(
                ppm_out, size=x.size()[2:],
                mode='bilinear', align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class UPerHead(nn.Module):
    def __init__(self, in_channels=(96, 192, 384, 768), num_classes=40,
                 channels=128, pool_scales=(1, 2, 3, 6),
                 norm_layer=nn.BatchNorm2d, dropout_ratio=0.1,
                 align_corners=False, **kwargs):
        super(UPerHead, self).__init__()
        self.in_channels = list(in_channels)
        self.channels = channels
        self.align_corners = align_corners

        self.psp_modules = PPM(
            pool_scales, self.in_channels[-1], self.channels, norm_layer, align_corners)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.in_channels[-1] + len(pool_scales) * self.channels,
                      self.channels, 3, padding=1, bias=False),
            norm_layer(self.channels),
            nn.ReLU(inplace=True),
        )

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_c in self.in_channels[:-1]:
            self.lateral_convs.append(nn.Sequential(
                nn.Conv2d(in_c, self.channels, 1, bias=False),
                norm_layer(self.channels),
                nn.ReLU(inplace=False),
            ))
            self.fpn_convs.append(nn.Sequential(
                nn.Conv2d(self.channels, self.channels, 3, padding=1, bias=False),
                norm_layer(self.channels),
                nn.ReLU(inplace=False),
            ))

        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(len(self.in_channels) * self.channels, self.channels,
                      3, padding=1, bias=False),
            norm_layer(self.channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.conv_seg = nn.Conv2d(self.channels, num_classes, kernel_size=1)

    def psp_forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        return self.bottleneck(torch.cat(psp_outs, dim=1))

    def forward(self, inputs):
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals.append(self.psp_forward(inputs))

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:],
                mode='bilinear', align_corners=self.align_corners)

        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:],
                mode='bilinear', align_corners=self.align_corners)

        output = self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))
        output = self.dropout(output)
        return self.conv_seg(output)


__all__ = ["UPerHead"]

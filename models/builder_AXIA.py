import torch.nn as nn
import torch.nn.functional as F

from .encoders import dual_encoder
from .decoders.SARD import SARD
from .decoders.fcnhead import FCNHead
from .decoders.MLPDecoder import DecoderHead as MLPDecoderHead
from .decoders.deeplabv3plus import DeepLabV3Plus
from engine.logger import get_logger

logger = get_logger()


class EncoderDecoder(nn.Module):
    def __init__(self, cfg, criterion=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.channels = [96, 192, 384, 768]
        self.norm_layer = norm_layer
        if cfg.backbone != 'dual_encoder':
            raise ValueError(f'Unsupported backbone: {cfg.backbone}')
        self.backbone = dual_encoder.swin_l_ccrl_s()
        channels = getattr(cfg, 'decoder_channels', 512)
        dropout = getattr(cfg, 'decoder_dropout', 0.1)
        decoder = cfg.decoder
        if decoder == 'SARD':
            self.decode_head = SARD(
                in_channels=self.channels, channels=channels, dropout_ratio=dropout,
                num_classes=cfg.num_classes, norm_layer=norm_layer, align_corners=False)
        elif decoder == 'MLPDecoder':
            self.decode_head = MLPDecoderHead(
                in_channels=self.channels, embed_dim=channels,
                num_classes=cfg.num_classes, norm_layer=norm_layer,
                dropout_ratio=dropout, align_corners=False)
        elif decoder == 'DeepLabV3Plus':
            self.decode_head = DeepLabV3Plus(
                in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer)
        elif decoder == 'FCNHead':
            self.decode_head = FCNHead(
                in_channels=self.channels, channels=channels, num_classes=cfg.num_classes,
                norm_layer=norm_layer, in_index=-1)
        else:
            raise ValueError(f'Unsupported decoder: {decoder}')
        aux_name = getattr(cfg, 'aux_head', 'FCNHead')
        if aux_name == 'FCNHead':
            self.aux_head = FCNHead(
                in_channels=self.channels, num_classes=cfg.num_classes,
                norm_layer=norm_layer, in_index=2)
        elif aux_name is None:
            self.aux_head = None
        else:
            raise ValueError(f'Unsupported aux_head: {aux_name}')
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss(
            ignore_index=getattr(cfg, 'background', 255))
        self.aux_rate = 0.4 if self.aux_head is not None else 0.0
        if cfg.pretrained_model:
            dual_encoder.load_swin_rgb_weights(self.backbone, cfg.pretrained_model)
        sar_pretrained = getattr(cfg, 'sar_pretrained_model', None)
        if sar_pretrained:
            dual_encoder.load_swin_sar_weights(self.backbone, sar_pretrained)
        self.backbone.freeze_rgb_backbone()
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f'AXIA parameters: total={total:,}, trainable={trainable:,}')

    def encode_decode(self, rgb, modal_x):
        features = self.backbone(rgb, modal_x)
        logits = self.decode_head(features)
        return F.interpolate(logits, size=rgb.shape[2:], mode='bilinear',
                             align_corners=False), features

    def forward(self, rgb, modal_x, label=None):
        logits, features = self.encode_decode(rgb, modal_x)
        if label is None:
            return logits
        loss = self.criterion(logits, label.long())
        if self.aux_head is not None:
            aux = self.aux_head(features)
            aux = F.interpolate(aux, size=rgb.shape[2:], mode='bilinear', align_corners=False)
            loss = loss + self.aux_rate * self.criterion(aux, label.long())
        return loss

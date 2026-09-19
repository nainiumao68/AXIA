"""AXIA model builder.

Assembles the AXIA architecture:
  - Encoder: AXIAEncoder (dual-parallel Swin-Large RGB, frozen + Swin-Tiny X,
    stages 3-4 frozen; DGA/SSA block-level adapters + LSGF stage fusion).
  - Decoder: SARD by default; FCNHead / UPerHead / MLPDecoder are available
    as ablation alternatives.
  - Auxiliary head (optional, training only): FCNHead by default; AFDAH is
    also available.

Config fields consumed (see the per-dataset config files):
  backbone            must be 'dual_encoder' (the AXIA dual-parallel encoder)
  pretrained_model    ImageNet-22k Swin-Large checkpoint for the RGB branch
  sar_pretrained_model  Swin-Tiny checkpoint for the X branch
  rgb_is_four_channel   set True for 4-channel RGBI input (default False)
  decoder             'SARD' | 'FCNHead' | 'UPerHead' | 'MLPDecoder'
  decoder_channels    decoder width (default 512)
  decoder_dropout     decoder dropout ratio (default 0.1)
  aux_head            'FCNHead' | 'AFDAH' | None
  aux_head_embed_dim  AFDAH internal width (default 128)
  aux_loss_weight     weight of the auxiliary CE loss (default 0.4)
  num_classes         segmentation classes
"""
import torch.nn as nn
import torch.nn.functional as F

from .encoders.axia_encoder import (
    AXIAEncoder,
    load_swin_rgb_weights,
    load_swin_sar_weights,
)
from .decoders.SARD import SARD
from .decoders.fcnhead import FCNHead
from .decoders.UPernet import UPerHead
from .decoders.MLPDecoder import MLPDecoder
from .decoders.AFDAH import AFDAH
from engine.logger import get_logger

logger = get_logger()


def _count_params(module, trainable_only=False):
    if module is None:
        return 0
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


class EncoderDecoder(nn.Module):
    def __init__(self, cfg=None,
                 criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255),
                 norm_layer=nn.BatchNorm2d):
        super(EncoderDecoder, self).__init__()
        self.channels = [96, 192, 384, 768]
        self.norm_layer = norm_layer

        backbone_name = getattr(cfg, 'backbone', 'dual_encoder')
        if backbone_name != 'dual_encoder':
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. "
                "AXIA supports exactly one backbone: 'dual_encoder'.")
        logger.info('Using backbone: AXIAEncoder - Swin-Large (frozen RGB) + '
                    'Swin-Tiny (X, stages 3-4 frozen), DGA+SSA adapters, LSGF fusion')

        if getattr(cfg, 'rgb_is_four_channel', False):
            logger.info('[backbone] 4-channel RGB input (RGBI) -> swin_in_chans=4')
            self.backbone = AXIAEncoder(norm_fuse=norm_layer, swin_in_chans=4)
        else:
            self.backbone = AXIAEncoder(norm_fuse=norm_layer)

        decoder_name = getattr(cfg, 'decoder', 'SARD')
        decoder_channels = getattr(cfg, 'decoder_channels', 512)
        decoder_kwargs = dict(
            in_channels=self.channels,
            channels=decoder_channels,
            dropout_ratio=getattr(cfg, 'decoder_dropout', 0.1),
            num_classes=cfg.num_classes,
            norm_layer=norm_layer,
            align_corners=False,
        )

        _DECODER_REGISTRY = {
            'SARD': lambda: SARD(pool_scales=(1, 2, 3, 6), **decoder_kwargs),
            'FCNHead': lambda: FCNHead(**decoder_kwargs),
            'UPerHead': lambda: UPerHead(pool_scales=(1, 2, 3, 6), **decoder_kwargs),
            'MLPDecoder': lambda: MLPDecoder(**decoder_kwargs),
        }
        if decoder_name not in _DECODER_REGISTRY:
            raise ValueError(
                f"Unknown decoder '{decoder_name}'. "
                f"Supported: {list(_DECODER_REGISTRY.keys())}")
        logger.info(f'Using {decoder_name}')
        self.decode_head = _DECODER_REGISTRY[decoder_name]()
        self._decoder_name = decoder_name

        aux_head_name = getattr(cfg, 'aux_head', 'FCNHead')
        if aux_head_name is None:
            self.aux_head = None
            logger.info('[Aux] Auxiliary head disabled')
        elif aux_head_name == 'FCNHead':
            self.aux_head = FCNHead(
                in_channels=tuple(self.channels),
                num_classes=cfg.num_classes,
                channels=256,
                dropout_ratio=0.1,
                norm_layer=norm_layer,
            )
            logger.info('[Aux] Using FCNHead auxiliary head')
        elif aux_head_name == 'AFDAH':
            self.aux_head = AFDAH(
                in_channels=tuple(self.channels),
                num_classes=cfg.num_classes,
                embed_dim=getattr(cfg, 'aux_head_embed_dim', 128),
            )
            logger.info('[Aux] Using AFDAH auxiliary head')
        else:
            raise ValueError(
                f"Unknown aux_head '{aux_head_name}'. "
                "AXIA supports 'FCNHead', 'AFDAH', or None.")

        decode_total = _count_params(self.decode_head, trainable_only=False)
        decode_trainable = _count_params(self.decode_head, trainable_only=True)
        logger.info(f'[Decoder] {decoder_name}: total={decode_total:,} '
                    f'trainable={decode_trainable:,}')
        if self.aux_head is not None:
            aux_total = _count_params(self.aux_head, trainable_only=False)
            aux_trainable = _count_params(self.aux_head, trainable_only=True)
            logger.info(f'[Aux] {aux_head_name}: total={aux_total:,} trainable={aux_trainable:,}')

        self.criterion = criterion
        self.aux_loss_weight = float(getattr(cfg, 'aux_loss_weight', 0.4))

        self.init_weights(cfg, pretrained=cfg.pretrained_model)

    def init_weights(self, cfg, pretrained=None):
        # RGB branch: ImageNet-22k Swin-Large.
        if pretrained:
            logger.info(f'Loading Swin-Large (RGB) pretrained: {pretrained}')
            load_swin_rgb_weights(self.backbone, pretrained)

        # X branch: Swin-Tiny.
        sar_pretrained = getattr(cfg, 'sar_pretrained_model', None)
        if sar_pretrained:
            logger.info(f'Loading Swin-Tiny (X) pretrained: {sar_pretrained}')
            load_swin_sar_weights(self.backbone, sar_pretrained)

        # Freeze the RGB branch and the configured X stages.
        self.backbone.freeze_rgb_backbone()
        frozen = sum(p.numel() for p in self.backbone.parameters() if not p.requires_grad)
        adapters = sum(p.numel() for n, p in self.backbone.named_parameters()
                       if ('dga' in n or 'ssa' in n) and p.requires_grad)
        logger.info(f'[Freeze] Backbone frozen: {frozen:,} params '
                    f'(DGA+SSA adapters remain trainable: {adapters:,})')

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f'[Freeze] Total: {total:,}  Trainable: {trainable:,}  '
                    f'Frozen: {total - trainable:,} ({(total - trainable) / total * 100:.1f}%)')

    def encode_decode(self, rgb, modal_x):
        orisize = rgb.shape
        x = self.backbone(rgb, modal_x)

        seg = self.decode_head(x)
        out = F.interpolate(seg, size=orisize[2:], mode='bilinear', align_corners=False)

        return out, x

    def forward(self, rgb, modal_x, label=None):
        out, encoder_feats = self.encode_decode(rgb, modal_x)

        if label is not None:
            loss = self.criterion(out, label.long())

            if self.aux_head is not None:
                if hasattr(self.aux_head, 'compute_loss'):
                    loss = loss + self.aux_head.compute_loss(
                        encoder_feats, label, rgb.shape[2:], self.criterion)
                else:
                    aux_logits = self.aux_head(encoder_feats)
                    aux_logits = F.interpolate(
                        aux_logits, size=rgb.shape[2:], mode='bilinear',
                        align_corners=False)
                    loss = loss + self.aux_loss_weight * self.criterion(
                        aux_logits, label.long())

            return loss
        return out

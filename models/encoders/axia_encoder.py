"""AXIA dual-parallel encoder.

RGB branch: Swin-Large (ImageNet-22k pretrained, fully frozen).
X branch:   Swin-Tiny (pretrained; stages 1-2 trainable, stages 3-4 frozen).

Three lightweight cross-modal fusion modules:
  1. DGA  (Discrepancy-Guided Adapter) - block-level, inserted after the
     window attention of each Swin-Large block.
  2. SSA  (Shared-Subspace Adapter)    - block-level, inserted after the FFN
     of each Swin-Large block; also updates the X tokens.
  3. LSGF (Light Stage-Gate Fusion)    - stage-level confidence-gated additive
     fusion (the ``FFMs`` module list).

Channel projection: Swin-Large dims [192, 384, 768, 1536] are projected to the
target dims [96, 192, 384, 768]; the Swin-Tiny X branch already matches them.

Feature flow per stage:
  1. The X Swin-Tiny stage completes all its blocks -> x_tokens.
  2. Each RGB Swin-Large block runs with DGA + SSA inserted:
     W-MSA/SW-MSA -> +residual -> DGA(x_rgb, x_tokens) ->
     LN -> FFN -> +residual -> SSA(x_rgb, x_tokens).
  3. The SSA-updated X tokens are carried into the next stage.
  4. Stage outputs are channel-projected and fused by LSGF.

Note: module attribute names (``swin_*``, ``aux_*``, ``adapters``, ``FFMs``,
``rgb_proj``, ``sar_proj``, ``aux_norms``) are part of the checkpoint format
and must not be renamed.
"""
import math
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
try:
    from timm.layers import trunc_normal_, DropPath, to_2tuple
except ImportError:  # older timm
    from timm.models.layers import trunc_normal_, DropPath, to_2tuple

from modules.DiscrepancyGuidedAdapter import DGA
from modules.SharedSubspaceAdapter import SSA


# ===================== Utilities =====================

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


# ===================== Shared Swin Components =====================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)
        attn = attn + relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)
        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        identity = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = identity + self.drop_path(x)
        return x


class SwinPatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W
        x = x.view(B, H, W, C)
        if (H % 2 == 1) or (W % 2 == 1):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x = torch.cat([x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :],
                       x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class SwinBasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=7, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x, H, W):
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0))
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        return x, H, W


class SwinPatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x


# ===================== Stage-Level Fusion (LSGF) =====================

def _spatial_norm(num_channels):
    num_groups = min(8, num_channels)
    while num_groups > 1 and num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class LightStageGateFusion(nn.Module):
    """Confidence-gated additive stage fusion (LSGF).

    The block-level adapters (DGA/SSA) have already cross-calibrated the two
    streams, so the only remaining job at the stage boundary is to collapse
    them into a single feature map with a minimal, data-dependent confidence
    correction on top of the strong baseline ``y = x_rgb + x_sar``:

        y = x_rgb + (1 + tanh(delta_c + delta_s)) * x_sar

    where ``delta_c`` is a per-channel correction from paired GAP descriptors
    and ``delta_s`` a per-spatial correction from the cosine agreement map.
    The multiplier lives in (0, 2) centred at 1, and both gates are
    zero-initialised so the module starts as pure additive fusion.
    """

    def __init__(self, dim, reduction=16):
        super().__init__()
        hidden = max(dim // reduction, 8)

        self.rgb_norm = _spatial_norm(dim)
        self.sar_norm = _spatial_norm(dim)

        self.channel_gate = nn.Sequential(
            nn.Linear(2 * dim, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, dim, bias=True),
        )
        self.spatial_gate = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True)

        self._zero_init_gates()

    def _zero_init_gates(self):
        last_linear = self.channel_gate[-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)
        nn.init.zeros_(self.spatial_gate.weight)
        nn.init.zeros_(self.spatial_gate.bias)
        first_linear = self.channel_gate[0]
        nn.init.kaiming_uniform_(first_linear.weight, a=5 ** 0.5)

    def forward(self, x_rgb, x_sar):
        if x_rgb.shape != x_sar.shape:
            raise ValueError(
                f"Shape mismatch: RGB {tuple(x_rgb.shape)} vs SAR {tuple(x_sar.shape)}."
            )

        r = self.rgb_norm(x_rgb)
        s = self.sar_norm(x_sar)

        gap = torch.cat(
            [
                F.adaptive_avg_pool2d(r, 1).flatten(1),
                F.adaptive_avg_pool2d(s, 1).flatten(1),
            ],
            dim=1,
        )
        delta_c = self.channel_gate(gap).unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)

        cos = F.cosine_similarity(r, s, dim=1, eps=1e-6).unsqueeze(1)  # (B, 1, H, W)
        delta_s = self.spatial_gate(cos)                               # (B, 1, H, W)

        weight = 1.0 + torch.tanh(delta_c + delta_s)                   # (B, C, H, W)
        return x_rgb + weight * x_sar


# ===================== AXIA Dual Encoder =====================

class AXIAEncoder(nn.Module):
    """Dual-parallel encoder: Swin-Large (RGB, frozen) + Swin-Tiny (X, partially frozen).

    The RGB branch is frozen via :meth:`freeze_rgb_backbone`; the X branch
    freezes the stages listed in ``sar_freeze_stages`` (zero-based).  With the
    default ``(2, 3)``, X stages 3-4 and their patch-merging transitions are
    frozen while stages 1-2 remain trainable.
    """

    # Zero-based X Swin stages to freeze; () = fully trainable X branch.
    sar_freeze_stages = (2, 3)
    sar_freeze_patch_embed = True

    def __init__(self,
                 # RGB Swin-Large
                 swin_pretrain_img_size=224,
                 swin_patch_size=4,
                 swin_embed_dim=192,
                 swin_depths=(2, 2, 18, 2),
                 swin_num_heads=(6, 12, 24, 48),
                 swin_window_size=7,
                 swin_mlp_ratio=4.,
                 swin_qkv_bias=True,
                 swin_qk_scale=None,
                 swin_drop_rate=0.,
                 swin_attn_drop_rate=0.,
                 swin_drop_path_rate=0.2,
                 swin_ape=False,
                 swin_patch_norm=True,
                 swin_use_checkpoint=False,
                 swin_in_chans=None,
                 # X Swin-Tiny
                 sar_patch_size=4,
                 sar_embed_dim=96,
                 sar_depths=(2, 2, 6, 2),
                 sar_num_heads=(3, 6, 12, 24),
                 sar_window_size=7,
                 sar_mlp_ratio=4.,
                 sar_qkv_bias=True,
                 sar_qk_scale=None,
                 sar_drop_rate=0.,
                 sar_attn_drop_rate=0.,
                 sar_drop_path_rate=0.2,
                 sar_ape=False,
                 sar_patch_norm=True,
                 sar_use_checkpoint=False,
                 # Shared
                 target_dims=(96, 192, 384, 768),
                 in_chans=3,
                 norm_layer=None,
                 norm_fuse=nn.BatchNorm2d):
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_stages = 4
        self.swin_ape = swin_ape
        self.sar_ape = sar_ape

        self.swin_dims = [int(swin_embed_dim * 2 ** i) for i in range(self.num_stages)]
        self.sar_dims = [int(sar_embed_dim * 2 ** i) for i in range(self.num_stages)]
        self.target_dims = list(target_dims)
        self.num_features = list(target_dims)

        _swin_in_chans = swin_in_chans if swin_in_chans is not None else in_chans

        # ===== Swin-Large (RGB, frozen) =====
        self.swin_patch_embed = SwinPatchEmbed(
            patch_size=swin_patch_size, in_chans=_swin_in_chans, embed_dim=swin_embed_dim,
            norm_layer=norm_layer if swin_patch_norm else None)

        if swin_ape:
            pretrain_img_size_t = to_2tuple(swin_pretrain_img_size)
            patch_size_t = to_2tuple(swin_patch_size)
            patches_resolution = [pretrain_img_size_t[0] // patch_size_t[0],
                                  pretrain_img_size_t[1] // patch_size_t[1]]
            self.swin_absolute_pos_embed = nn.Parameter(
                torch.zeros(1, swin_embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.swin_absolute_pos_embed, std=.02)

        self.swin_pos_drop = nn.Dropout(p=swin_drop_rate)

        swin_dpr = [x.item() for x in torch.linspace(0, swin_drop_path_rate, sum(swin_depths))]
        self.swin_layers = nn.ModuleList()
        self.swin_downsamples = nn.ModuleList()
        for i_layer in range(self.num_stages):
            self.swin_layers.append(SwinBasicLayer(
                dim=self.swin_dims[i_layer],
                depth=swin_depths[i_layer],
                num_heads=swin_num_heads[i_layer],
                window_size=swin_window_size,
                mlp_ratio=swin_mlp_ratio,
                qkv_bias=swin_qkv_bias,
                qk_scale=swin_qk_scale,
                drop=swin_drop_rate,
                attn_drop=swin_attn_drop_rate,
                drop_path=swin_dpr[sum(swin_depths[:i_layer]):sum(swin_depths[:i_layer + 1])],
                norm_layer=norm_layer,
                use_checkpoint=swin_use_checkpoint))
            if i_layer < self.num_stages - 1:
                self.swin_downsamples.append(
                    SwinPatchMerging(dim=self.swin_dims[i_layer], norm_layer=norm_layer))

        # ===== DGA + SSA adapters (one pair per Swin-Large block per stage) =====
        self.adapters = nn.ModuleList()
        for stage_idx in range(self.num_stages):
            stage_adapters = nn.ModuleList()
            rgb_dim = self.swin_dims[stage_idx]
            sar_dim = self.sar_dims[stage_idx]
            for _ in range(swin_depths[stage_idx]):
                stage_adapters.append(nn.ModuleDict({
                    'dga': DGA(in_dim=rgb_dim, sar_dim=sar_dim),
                    'ssa': SSA(in_dim=rgb_dim, sar_dim=sar_dim),
                }))
            self.adapters.append(stage_adapters)

        # ===== Channel projection: Swin-Large dims -> target dims =====
        self.rgb_proj = nn.ModuleList()
        for s_d, t_d in zip(self.swin_dims, target_dims):
            if s_d != t_d:
                self.rgb_proj.append(nn.Sequential(
                    nn.Conv2d(s_d, t_d, 1, bias=False),
                    nn.BatchNorm2d(t_d),
                ))
            else:
                self.rgb_proj.append(nn.Identity())

        # ===== Swin-Tiny (X branch) =====
        self.aux_patch_embed = SwinPatchEmbed(
            patch_size=sar_patch_size,
            in_chans=in_chans, embed_dim=sar_embed_dim,
            norm_layer=norm_layer if sar_patch_norm else None)

        if sar_ape:
            pretrain_img_size_t = to_2tuple(swin_pretrain_img_size)
            patch_size_t = to_2tuple(swin_patch_size)
            patches_resolution = [pretrain_img_size_t[0] // patch_size_t[0],
                                  pretrain_img_size_t[1] // patch_size_t[1]]
            self.aux_absolute_pos_embed = nn.Parameter(
                torch.zeros(1, sar_embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.aux_absolute_pos_embed, std=.02)

        self.aux_pos_drop = nn.Dropout(p=sar_drop_rate)

        sar_dpr = [x.item() for x in torch.linspace(0, sar_drop_path_rate, sum(sar_depths))]
        self.aux_layers = nn.ModuleList()
        self.aux_downsamples = nn.ModuleList()
        for i_layer in range(self.num_stages):
            self.aux_layers.append(SwinBasicLayer(
                dim=self.sar_dims[i_layer],
                depth=sar_depths[i_layer],
                num_heads=sar_num_heads[i_layer],
                window_size=sar_window_size,
                mlp_ratio=sar_mlp_ratio,
                qkv_bias=sar_qkv_bias,
                qk_scale=sar_qk_scale,
                drop=sar_drop_rate,
                attn_drop=sar_attn_drop_rate,
                drop_path=sar_dpr[sum(sar_depths[:i_layer]):sum(sar_depths[:i_layer + 1])],
                norm_layer=norm_layer,
                use_checkpoint=sar_use_checkpoint))
            if i_layer < self.num_stages - 1:
                self.aux_downsamples.append(
                    SwinPatchMerging(dim=self.sar_dims[i_layer], norm_layer=norm_layer))

        # Per-stage LayerNorm on the X tokens, applied before fusion.
        self.aux_norms = nn.ModuleList([
            norm_layer(self.sar_dims[i]) for i in range(self.num_stages)])

        # X projection is identity when the branch dims already match the target dims.
        self.sar_proj = nn.ModuleList()
        for s_d, t_d in zip(self.sar_dims, target_dims):
            if s_d != t_d:
                self.sar_proj.append(nn.Sequential(
                    nn.Conv2d(s_d, t_d, 1, bias=False),
                    nn.BatchNorm2d(t_d),
                ))
            else:
                self.sar_proj.append(nn.Identity())

        # ===== Stage-level fusion =====
        self.FFMs = nn.ModuleList([
            LightStageGateFusion(dim=target_dims[i]) for i in range(4)])

        self.apply(self._init_weights)

    # ------------------------------------------------------------------

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # ------------------------------------------------------------------
    # Freezing

    @staticmethod
    def _set_trainable(module, trainable):
        module.train(trainable)
        for param in module.parameters():
            param.requires_grad = trainable

    def _freeze_sar_parts(self):
        """Freeze the X Swin-Tiny stages listed in ``sar_freeze_stages``.

        When stage i is frozen, its patch-merging transition is frozen too,
        matching how official Swin checkpoints store ``layers.i.downsample``.
        """
        if not self.sar_freeze_stages:
            return
        if self.sar_freeze_patch_embed and 0 in self.sar_freeze_stages:
            self._set_trainable(self.aux_patch_embed, False)
            self.aux_pos_drop.eval()
            if self.sar_ape:
                self.aux_absolute_pos_embed.requires_grad = False
        for stage_idx in self.sar_freeze_stages:
            if not 0 <= stage_idx < len(self.aux_layers):
                raise ValueError(f"Invalid X Swin stage index: {stage_idx}")
            self._set_trainable(self.aux_layers[stage_idx], False)
            if stage_idx < len(self.aux_downsamples):
                self._set_trainable(self.aux_downsamples[stage_idx], False)

    def freeze_rgb_backbone(self):
        """Freeze the full RGB Swin-Large branch, then apply X stage freezing."""
        for name, param in self.named_parameters():
            if name.startswith('swin_'):
                param.requires_grad = False
        self._freeze_sar_parts()

    def train(self, mode=True):
        super().train(mode)
        # Keep frozen X stages in eval mode (BN/dropout) across train() calls.
        self._freeze_sar_parts()
        return self

    # ------------------------------------------------------------------
    # Per-block Swin-Large execution with DGA + SSA

    def _run_swin_block(self, blk, x, sar_tokens, attn_mask, H, W, dga, ssa):
        """Split SwinTransformerBlock forward; insert DGA after attention and
        SSA after the FFN.  x: (B, N, C) tokens; sar_tokens: (B, N, C_sar)."""
        B, L, C = x.shape

        shortcut = x
        x = blk.norm1(x)
        x = x.view(B, H, W, C)
        pad_r = (blk.window_size - W % blk.window_size) % blk.window_size
        pad_b = (blk.window_size - H % blk.window_size) % blk.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape
        if blk.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-blk.shift_size, -blk.shift_size), dims=(1, 2))
            mask = attn_mask
        else:
            shifted_x = x
            mask = None
        x_windows = window_partition(shifted_x, blk.window_size)
        x_windows = x_windows.view(-1, blk.window_size * blk.window_size, C)
        attn_windows = blk.attn(x_windows, mask=mask)
        attn_windows = attn_windows.view(-1, blk.window_size, blk.window_size, C)
        shifted_x = window_reverse(attn_windows, blk.window_size, Hp, Wp)
        if blk.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(blk.shift_size, blk.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = shortcut + blk.drop_path(x)

        # DGA after attention
        x = dga(x, sar_tokens, hw_shapes=(H, W))

        identity = x
        x = blk.norm2(x)
        x = blk.mlp(x)
        x = identity + blk.drop_path(x)

        # SSA after FFN
        x, sar_tokens = ssa(x, sar_tokens, hw_shapes=(H, W))

        return x, sar_tokens

    # ------------------------------------------------------------------

    @staticmethod
    def _compute_attn_mask(layer, H, W, device):
        Hp = int(np.ceil(H / layer.window_size)) * layer.window_size
        Wp = int(np.ceil(W / layer.window_size)) * layer.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
        h_slices = (slice(0, -layer.window_size),
                    slice(-layer.window_size, -layer.shift_size),
                    slice(-layer.shift_size, None))
        w_slices = (slice(0, -layer.window_size),
                    slice(-layer.window_size, -layer.shift_size),
                    slice(-layer.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, layer.window_size)
        mask_windows = mask_windows.view(-1, layer.window_size * layer.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0))

    # ------------------------------------------------------------------

    def forward_features(self, x_rgb, x_aux):
        out = []
        B = x_rgb.shape[0]

        # ---- RGB Swin-Large patch embedding ----
        swin_x = self.swin_patch_embed(x_rgb)
        Wh, Ww = swin_x.size(2), swin_x.size(3)
        swin_x = swin_x.flatten(2).transpose(1, 2)
        if self.swin_ape:
            abs_pe = F.interpolate(self.swin_absolute_pos_embed,
                                   size=(Wh, Ww), mode='bicubic')
            swin_x = swin_x + abs_pe.flatten(2).transpose(1, 2)
        swin_x = self.swin_pos_drop(swin_x)

        # ---- X Swin-Tiny patch embedding ----
        aux_x = self.aux_patch_embed(x_aux)
        aWh, aWw = aux_x.size(2), aux_x.size(3)
        aux_x = aux_x.flatten(2).transpose(1, 2)   # (B, N_x, C_x)
        if self.sar_ape:
            abs_pe_sar = F.interpolate(self.aux_absolute_pos_embed,
                                       size=(aWh, aWw), mode='bicubic')
            aux_x = aux_x + abs_pe_sar.flatten(2).transpose(1, 2)
        aux_x = self.aux_pos_drop(aux_x)

        for i in range(self.num_stages):
            H, W = Wh, Ww

            # --- 1) X Swin-Tiny: complete this stage first ---
            aux_layer = self.aux_layers[i]
            aux_attn_mask = self._compute_attn_mask(aux_layer, aWh, aWw, aux_x.device)
            for blk in aux_layer.blocks:
                blk.H, blk.W = aWh, aWw
                aux_x = blk(aux_x, aux_attn_mask)

            Hu, Wu = aWh, aWw
            sar_tokens = aux_x   # (B, N_x, C_x)

            # --- 2) RGB Swin-Large: run each block with DGA + SSA ---
            layer = self.swin_layers[i]
            attn_mask = self._compute_attn_mask(layer, H, W, swin_x.device)
            for blk_idx, blk in enumerate(layer.blocks):
                blk.H, blk.W = H, W
                dga = self.adapters[i][blk_idx]['dga']
                ssa = self.adapters[i][blk_idx]['ssa']
                swin_x, sar_tokens = self._run_swin_block(
                    blk, swin_x, sar_tokens, attn_mask, H, W, dga, ssa)

            # Keep the SSA-updated X tokens as the next-stage X input.
            aux_x = sar_tokens

            # Channel-project the RGB output.
            swin_spatial = (swin_x.view(B, H, W, self.swin_dims[i])
                            .permute(0, 3, 1, 2).contiguous())
            x_proj = self.rgb_proj[i](swin_spatial)     # (B, target_dim, H, W)

            # LayerNorm + spatial reshape of the X tokens for fusion.
            sar_normed = self.aux_norms[i](sar_tokens)
            sar_spatial = (sar_normed.reshape(B, Hu, Wu, self.sar_dims[i])
                           .permute(0, 3, 1, 2).contiguous())
            sar_spatial = self.sar_proj[i](sar_spatial)

            # --- 3) Stage-level fusion ---
            out.append(self.FFMs[i](x_proj, sar_spatial))

            # Swin-Large downsampling for the next stage.
            if i < self.num_stages - 1:
                swin_x = self.swin_downsamples[i](swin_x, H, W)
                Wh, Ww = (H + 1) // 2, (W + 1) // 2

            # Swin-Tiny downsampling for the next stage.
            if i < self.num_stages - 1:
                aux_x = self.aux_downsamples[i](aux_x, Hu, Wu)
                aWh, aWw = (Hu + 1) // 2, (Wu + 1) // 2

        return tuple(out)

    def forward(self, x_rgb, x_aux):
        return self.forward_features(x_rgb, x_aux)


# ===================== Weight Loading =====================

def load_swin_rgb_weights(model, pretrained_path):
    """Load Swin-Large pretrained weights into the ``swin_*`` modules."""
    t_start = time.time()
    raw = torch.load(pretrained_path, map_location='cpu')
    if 'model' in raw:
        raw = raw['model']
    elif 'state_dict' in raw:
        bd = {}
        for k, v in raw['state_dict'].items():
            if k.startswith('backbone.'):
                bd[k[9:]] = v
        raw = bd

    cleaned = {(k[9:] if k.startswith('backbone.') else k): v for k, v in raw.items()}

    swin_state = {}
    for k, v in cleaned.items():
        if 'head' in k or 'attn_mask' in k:
            continue
        if k.startswith('patch_embed.'):
            swin_state['swin_patch_embed.' + k[len('patch_embed.'):]] = v
        elif k.startswith('layers.'):
            parts = k.split('.', 3)
            layer_idx = parts[1]
            rest = parts[2] if len(parts) > 2 else ''
            remaining = parts[3] if len(parts) > 3 else ''
            if rest == 'downsample':
                swin_state[f'swin_downsamples.{layer_idx}.{remaining}'] = v
            else:
                tail = f'.{remaining}' if remaining else ''
                swin_state[f'swin_layers.{layer_idx}.{rest}{tail}'] = v

    # Handle in_chans mismatch for patch_embed (e.g. 3-channel pretrained -> 4-channel RGBI).
    patch_key = 'swin_patch_embed.proj.weight'
    if patch_key in swin_state:
        pretrained_w = swin_state[patch_key]
        model_w = model.state_dict().get(patch_key)
        if model_w is not None and model_w.shape[1] != pretrained_w.shape[1]:
            n_extra = model_w.shape[1] - pretrained_w.shape[1]
            if n_extra > 0:
                extra = pretrained_w.mean(dim=1, keepdim=True).expand(-1, n_extra, -1, -1)
                swin_state[patch_key] = torch.cat([pretrained_w, extra], dim=1)
                print(f"  [RGB patch_embed] Inflated in_chans: "
                      f"{pretrained_w.shape[1]} -> {model_w.shape[1]}")

    msg = model.load_state_dict(swin_state, strict=False)
    t_end = time.time()
    expected_miss = [
        k for k in msg.missing_keys
        if k.startswith('aux_') or 'adapters' in k or 'FFMs' in k
        or 'rgb_proj' in k or 'sar_proj' in k
    ]
    other_miss = [k for k in msg.missing_keys if k not in expected_miss]
    print(f"[RGB Swin-L] Loaded from {pretrained_path} ({t_end - t_start:.2f}s)")
    print(f"  Missing backbone: {len(other_miss)}, "
          f"Non-backbone missing (expected): {len(expected_miss)}, "
          f"Unexpected: {len(msg.unexpected_keys)}")
    if other_miss:
        print(f"  Backbone missing (first 5): {other_miss[:5]}")
    return model


def load_swin_sar_weights(model, pretrained_path):
    """Load Swin-Tiny pretrained weights into the X branch (``aux_*`` modules)."""
    t_start = time.time()
    try:
        raw = torch.load(pretrained_path, map_location='cpu', weights_only=True)
    except Exception:
        raw = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    if 'model' in raw:
        raw = raw['model']
    elif 'state_dict' in raw:
        bd = {}
        for k, v in raw['state_dict'].items():
            if k.startswith('backbone.'):
                bd[k[9:]] = v
        raw = bd

    aux_state = {}
    for k, v in raw.items():
        if 'head' in k or 'attn_mask' in k:
            continue
        if k.startswith('patch_embed.'):
            aux_state['aux_patch_embed.' + k[len('patch_embed.'):]] = v
        elif k.startswith('layers.'):
            parts = k.split('.', 3)
            layer_idx = parts[1]
            rest = parts[2] if len(parts) > 2 else ''
            remaining = parts[3] if len(parts) > 3 else ''
            if rest == 'downsample':
                aux_state[f'aux_downsamples.{layer_idx}.{remaining}'] = v
            else:
                tail = f'.{remaining}' if remaining else ''
                aux_state[f'aux_layers.{layer_idx}.{rest}{tail}'] = v

    # Handle in_chans mismatch for the X patch_embed (e.g. 3-channel pretrained -> 1-channel SAR).
    patch_key = 'aux_patch_embed.proj.weight'
    if patch_key in aux_state:
        pretrained_w = aux_state[patch_key]
        model_w = model.state_dict().get(patch_key)
        if model_w is not None and model_w.shape[1] != pretrained_w.shape[1]:
            c_pre = pretrained_w.shape[1]
            c_model = model_w.shape[1]
            if c_model < c_pre:
                aux_state[patch_key] = pretrained_w.mean(dim=1, keepdim=True).expand(
                    -1, c_model, -1, -1)
            else:
                extra = pretrained_w.mean(dim=1, keepdim=True).expand(-1, c_model - c_pre, -1, -1)
                aux_state[patch_key] = torch.cat([pretrained_w, extra], dim=1)
            print(f"  [X patch_embed] Adapted in_chans: {c_pre} -> {c_model}")

    model_state = model.state_dict()
    compatible_aux_state = {}
    skipped_shape = []
    for k, v in aux_state.items():
        model_v = model_state.get(k)
        if model_v is None or model_v.shape == v.shape:
            compatible_aux_state[k] = v
        else:
            skipped_shape.append((k, tuple(v.shape), tuple(model_v.shape)))

    msg = model.load_state_dict(compatible_aux_state, strict=False)
    t_end = time.time()
    expected_miss = [
        k for k in msg.missing_keys
        if k.startswith('swin_') or 'adapters' in k or 'FFMs' in k
        or 'rgb_proj' in k or 'sar_proj' in k or 'aux_norms' in k
    ]
    other_miss = [k for k in msg.missing_keys if k not in expected_miss]
    print(f"[X Swin-T] Loaded from {pretrained_path} ({t_end - t_start:.2f}s)")
    print(f"  Missing X backbone: {len(other_miss)}, "
          f"Non-backbone missing (expected): {len(expected_miss)}, "
          f"Unexpected: {len(msg.unexpected_keys)}, "
          f"Shape-skipped: {len(skipped_shape)}")
    if other_miss:
        print(f"  X backbone missing (first 5): {other_miss[:5]}")
    if skipped_shape:
        print(f"  X shape-skipped (first 5): {skipped_shape[:5]}")
    return model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AXIAEncoder(norm_fuse=nn.BatchNorm2d).to(device)
    model.freeze_rgb_backbone()
    rgb = torch.randn(1, 3, 256, 256).to(device)
    modal_x = torch.randn(1, 3, 256, 256).to(device)
    outs = model(rgb, modal_x)
    for i, o in enumerate(outs):
        print(f'Stage {i}: {o.shape}')
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fusion = sum(p.numel() for n, p in model.named_parameters()
                 if 'dga' in n or 'ssa' in n or 'FFMs' in n)
    print(f'Total: {total:,}  Trainable: {trainable:,}  Adapters+Fusion: {fusion:,}')

"""Data loading and preprocessing pipelines for RGB-X semantic segmentation.

``TrainPre`` / ``ValPre`` are callables executed inside
:class:`dataloader.RGBXDataset.RGBXDataset`; ``get_train_loader`` and
``get_val_dataset`` are the entry points used by the training / evaluation
scripts.
"""
import random

import cv2
import numpy as np
import torch
from torch.utils import data

from utils.transforms import generate_random_crop_pos, random_crop_pad_to_shape, normalize
from dataloader.RGBXDataset import RGBXDataset


def make_data_setting(config):
    """Assemble the RGBXDataset setting dict from a config object."""
    return {
        'rgb_root': config.rgb_root_folder,
        'rgb_format': config.rgb_format,
        'gt_root': config.gt_root_folder,
        'gt_format': config.gt_format,
        'transform_gt': config.gt_transform,
        'num_classes': getattr(config, 'num_classes', None),
        'ignore_index': getattr(config, 'background', 255),
        'label_mapping': getattr(config, 'label_mapping', None),
        'x_root': config.x_root_folder,
        'x_format': config.x_format,
        'x_single_channel': config.x_is_single_channel,
        'rgb_is_four_channel': getattr(config, 'rgb_is_four_channel', False),
        'class_names': getattr(config, 'class_names', None),
        'train_source': config.train_source,
        'eval_source': config.eval_source,
        'dataset_name': getattr(config, 'dataset_name', ''),
        'gt_color_palette': getattr(config, 'gt_color_palette', None),
    }


def _get_norm_stats(config, prefix):
    return getattr(config, f'{prefix}_norm_mean'), getattr(config, f'{prefix}_norm_std')


def _random_mirror(rgb, gt, modal_x):
    if random.random() >= 0.5:
        rgb = cv2.flip(rgb, 1)
        gt = cv2.flip(gt, 1)
        modal_x = cv2.flip(modal_x, 1)
    return rgb, gt, modal_x


def _random_vflip(rgb, gt, modal_x, p=0.5):
    """Vertical flip.  Only sensible for near-nadir aerial / ortho imagery;
    enable via ``config.train_vflip = True``."""
    if random.random() < p:
        rgb = cv2.flip(rgb, 0)
        gt = cv2.flip(gt, 0)
        modal_x = cv2.flip(modal_x, 0)
    return rgb, gt, modal_x


def _random_rot90(rgb, gt, modal_x):
    """Random 90/180/270-degree rotation for rotation-invariant overhead
    imagery; enable via ``config.train_rot90 = True``.  ``cv2.rotate`` is a
    pure transpose/flip, so labels stay lossless."""
    k = random.randint(0, 3)  # 0: identity, 1: 90, 2: 180, 3: 270 degrees
    if k == 0:
        return rgb, gt, modal_x
    code = {1: cv2.ROTATE_90_CLOCKWISE,
            2: cv2.ROTATE_180,
            3: cv2.ROTATE_90_COUNTERCLOCKWISE}[k]
    rgb = cv2.rotate(rgb, code)
    gt = cv2.rotate(gt, code)
    modal_x = cv2.rotate(modal_x, code)
    return rgb, gt, modal_x


def _random_photometric(rgb, brightness=0.2, contrast=0.2, saturation=0.1, p=0.5):
    """Light photometric jitter on RGB only; the auxiliary modality keeps its
    physical values.  Enable via ``config.train_photometric = True``."""
    if random.random() >= p:
        return rgb
    img = rgb.astype(np.float32)
    if brightness > 0:
        img = img * (1.0 + random.uniform(-brightness, brightness))
    if contrast > 0:
        c = 1.0 + random.uniform(-contrast, contrast)
        mean = img.mean(axis=(0, 1), keepdims=True)
        img = (img - mean) * c + mean
    if saturation > 0:
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] *= (1.0 + random.uniform(-saturation, saturation))
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def _random_scale(rgb, gt, modal_x, scales):
    scale = random.choice(scales)
    sh = int(rgb.shape[0] * scale)
    sw = int(rgb.shape[1] * scale)
    rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    modal_x = cv2.resize(modal_x, (sw, sh), interpolation=cv2.INTER_LINEAR)
    return rgb, gt, modal_x, scale


def _resize_triplet(rgb, gt, modal_x, hw):
    """Resize the triplet to ``(H, W)``; NEAREST for the label map."""
    target_h, target_w = hw
    rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    modal_x = cv2.resize(modal_x, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return rgb, gt, modal_x


def _parse_resize_hw(config):
    """Read ``config.input_resize_hw``; accepts ``(h, w)`` / ``[h, w]`` / int;
    ``None`` disables the input resize."""
    v = getattr(config, 'input_resize_hw', None)
    if v is None:
        return None
    if isinstance(v, int):
        return (v, v)
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return (int(v[0]), int(v[1]))
    raise ValueError(f"config.input_resize_hw has an invalid format: {v!r}")


class TrainPre(object):
    """Training preprocessor: optional input resize -> mirror/flip/rot90/
    photometric augmentation -> random scale -> normalize -> random crop to
    ``(config.image_height, config.image_width)``."""

    def __init__(self, config):
        self.config = config
        self.rgb_norm_mean, self.rgb_norm_std = _get_norm_stats(config, 'rgb')
        self.x_norm_mean, self.x_norm_std = _get_norm_stats(config, 'x')
        self.use_rot90 = bool(getattr(config, 'train_rot90', False))
        self.use_vflip = bool(getattr(config, 'train_vflip', False))
        self.use_photometric = bool(getattr(config, 'train_photometric', False))
        self.resize_hw = _parse_resize_hw(config)

    def __call__(self, rgb, gt, modal_x):
        if self.resize_hw is not None:
            rgb, gt, modal_x = _resize_triplet(rgb, gt, modal_x, self.resize_hw)

        rgb, gt, modal_x = _random_mirror(rgb, gt, modal_x)

        if self.use_vflip:
            rgb, gt, modal_x = _random_vflip(rgb, gt, modal_x)

        if self.use_rot90:
            rgb, gt, modal_x = _random_rot90(rgb, gt, modal_x)

        if self.use_photometric:
            rgb = _random_photometric(rgb)

        if self.config.train_scale_array is not None:
            rgb, gt, modal_x, _ = _random_scale(rgb, gt, modal_x, self.config.train_scale_array)

        rgb = normalize(rgb, self.rgb_norm_mean, self.rgb_norm_std)
        modal_x = normalize(modal_x, self.x_norm_mean, self.x_norm_std)

        crop_size = (self.config.image_height, self.config.image_width)
        crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)
        p_rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
        p_modal_x, _ = random_crop_pad_to_shape(modal_x, crop_pos, crop_size, 0)

        p_rgb = p_rgb.transpose(2, 0, 1)
        p_modal_x = p_modal_x.transpose(2, 0, 1)

        return p_rgb, p_gt, p_modal_x


class ValPre(object):
    """Validation preprocessor: optional input resize only.  Normalization is
    left to the evaluator so that raw pixels stay available for visualization."""

    def __init__(self, config):
        self.resize_hw = _parse_resize_hw(config)

    def __call__(self, rgb, gt, modal_x):
        if self.resize_hw is not None:
            rgb, gt, modal_x = _resize_triplet(rgb, gt, modal_x, self.resize_hw)
        return rgb, gt, modal_x


def get_train_loader(config):
    train_preprocess = TrainPre(config)
    total_samples = config.batch_size * config.niters_per_epoch
    train_dataset = RGBXDataset(make_data_setting(config), 'train',
                                train_preprocess, total_samples)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=True,
        pin_memory=True,
    )
    return train_loader


def get_val_dataset(config):
    return RGBXDataset(make_data_setting(config), 'val', ValPre(config))

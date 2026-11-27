import cv2
import torch
from torch.utils import data
import random
from utils.transforms import generate_random_crop_pos, random_crop_pad_to_shape, normalize
from dataloader.RGBXDataset import RGBXDataset


def _get_norm_stats(config, prefix):
    return getattr(config, f'{prefix}_norm_mean'), getattr(config, f'{prefix}_norm_std')


def _random_mirror(rgb, gt, modal_x):
    if random.random() >= 0.5:
        rgb = cv2.flip(rgb, 1)
        gt = cv2.flip(gt, 1)
        modal_x = cv2.flip(modal_x, 1)
    return (rgb, gt, modal_x)


def _random_scale(rgb, gt, modal_x, scales):
    scale = random.choice(scales)
    sh = int(rgb.shape[0] * scale)
    sw = int(rgb.shape[1] * scale)
    rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    modal_x = cv2.resize(modal_x, (sw, sh), interpolation=cv2.INTER_LINEAR)
    return (rgb, gt, modal_x, scale)


class TrainPreUnified(object):
    def __init__(self, config):
        self.config = config
        self.rgb_norm_mean, self.rgb_norm_std = _get_norm_stats(config, 'rgb')
        self.x_norm_mean, self.x_norm_std = _get_norm_stats(config, 'x')

    def __call__(self, rgb, gt, modal_x):
        rgb, gt, modal_x = _random_mirror(rgb, gt, modal_x)

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


def get_train_loader(engine, config):
    data_setting = {
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
        'train_source': config.train_source,
        'eval_source': config.eval_source,
        'gt_color_palette': getattr(config, 'gt_color_palette', None),
    }

    train_preprocess = TrainPreUnified(config)
    total_samples = config.batch_size * config.niters_per_epoch
    train_dataset = RGBXDataset(data_setting, "train", train_preprocess, total_samples)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=is_shuffle,
        pin_memory=True,
        sampler=train_sampler
    )

    return train_loader, train_sampler

"""Validation during training: mIoU tracking, best-model detection and
optional visualization dumps."""
import os
import random

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.metric import hist_info, compute_score
from utils.pyt_utils import ensure_dir
from utils.visualize import build_class_colors, colorize_mask
from dataloader.loader import get_val_dataset
from engine.logger import get_logger

logger = get_logger()


class ValidationHelper:
    def __init__(self, config, network, devices):
        self.config = config
        self.network = network
        self.devices = devices
        self.best_miou = 0.0
        self.best_epoch = 0
        self.rgb_norm_mean = config.rgb_norm_mean
        self.rgb_norm_std = config.rgb_norm_std
        self.x_norm_mean = config.x_norm_mean
        self.x_norm_std = config.x_norm_std

        self.dataset = get_val_dataset(config)

        self.val_loader = None
        if getattr(config, 'eval_batch_size', 1) > 1:
            from torch.utils.data import DataLoader
            self.val_loader = DataLoader(
                self.dataset,
                batch_size=config.eval_batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                drop_last=False,
            )

        ensure_dir(os.path.join(config.root_dir, 'val_results'))

    # ------------------------------------------------------------------
    # Preprocessing helpers (also used by test.py)

    def normalize(self, img, mean, std):
        img = img.astype(np.float32) / 255.0
        mean = np.asarray(mean)
        std = np.asarray(std)
        if img.ndim == 3:
            mean = mean.reshape(1, 1, -1)
            std = std.reshape(1, 1, -1)
        return (img - mean) / std

    def process_image_rgbX(self, img, modal_x):
        """Normalize an HWC RGB image and its X modality map, return CHW."""
        p_img = self.normalize(img, self.rgb_norm_mean, self.rgb_norm_std).transpose(2, 0, 1)
        if modal_x.ndim == 2:
            p_modal_x = self.normalize(modal_x, 0, 1)[np.newaxis, ...]
        else:
            p_modal_x = self.normalize(modal_x, self.x_norm_mean, self.x_norm_std).transpose(2, 0, 1)
        return p_img, p_modal_x

    # ------------------------------------------------------------------

    @staticmethod
    def _to_hwc_numpy(t):
        if isinstance(t, torch.Tensor):
            t = t.numpy()
        if t.ndim == 3 and t.shape[0] == 3:
            t = t.transpose(1, 2, 0)
        return t

    def direct_eval_rgbX(self, img, modal_x, model, device):
        """Single-sample full-resolution evaluation."""
        model.eval()
        img = self._to_hwc_numpy(img)
        modal_x = self._to_hwc_numpy(modal_x)
        ori_rows, ori_cols = img.shape[:2]

        p_img, p_modal_x = self.process_image_rgbX(img, modal_x)
        input_data = torch.from_numpy(np.ascontiguousarray(p_img[None])).float().to(device)
        input_modal_x = torch.from_numpy(np.ascontiguousarray(p_modal_x[None])).float().to(device)

        with torch.no_grad():
            score = model(input_data, input_modal_x)
            if isinstance(score, (list, tuple)):
                score = score[0]
            # Optional horizontal-flip TTA, controlled by config.eval_flip.
            if bool(getattr(self.config, 'eval_flip', False)):
                score_flip = model(input_data.flip(-1), input_modal_x.flip(-1))
                if isinstance(score_flip, (list, tuple)):
                    score_flip = score_flip[0]
                score = (score + score_flip.flip(-1)) * 0.5

            score = score.squeeze(0).permute(1, 2, 0).cpu().numpy()

        if score.shape[0] != ori_rows or score.shape[1] != ori_cols:
            score = cv2.resize(score, (ori_cols, ori_rows), interpolation=cv2.INTER_LINEAR)
        return score.argmax(2)

    # ------------------------------------------------------------------

    def validate(self, epoch, model):
        """Run validation and return ``(mIoU, is_best)``."""
        logger.info(f"Validating epoch {epoch} ...")
        model.eval()
        device = self.devices[0]

        self.hist = np.zeros((self.config.num_classes, self.config.num_classes))
        correct = 0
        labeled = 0

        # Randomly pick up to 20 samples for visualization.
        val_indices = list(range(len(self.dataset)))
        random.shuffle(val_indices)
        vis_indices = val_indices[:min(20, len(val_indices))]
        vis_images = []

        try:
            with torch.no_grad():
                if self.val_loader is not None:
                    labeled, correct, vis_images = self._validate_batched(
                        epoch, model, device, vis_indices)
                else:
                    labeled, correct, vis_images = self._validate_single(
                        epoch, model, device, vis_indices)

            if labeled == 0 or np.sum(self.hist) == 0:
                logger.warning("No valid labeled pixels during validation; skipped.")
                return 0.0, False

            n_eval = getattr(self.config, 'num_eval_classes', None)
            iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(
                self.hist, correct, labeled, num_eval_classes=n_eval)

            if np.isnan(mean_IoU) or np.isinf(mean_IoU):
                logger.warning("Invalid mIoU (NaN/Inf); treating as 0.")
                mean_IoU = 0.0
            if np.isnan(pixel_acc) or np.isinf(pixel_acc):
                pixel_acc = 0.0

            result_line = self.print_results(iou, mean_IoU, freq_IoU, mean_pixel_acc, pixel_acc)
            logger.info(f'Validation Epoch {epoch}:\n{result_line}')

            is_best = mean_IoU > self.best_miou
            if is_best:
                self.best_miou = mean_IoU
                self.best_epoch = epoch
                logger.info(f'New best model! mIoU: {mean_IoU:.4f}')
                if vis_images:
                    self.save_visualizations(vis_images, epoch)

            return mean_IoU, is_best

        except Exception as e:
            logger.error(f"Error during validation: {e}")
            return 0.0, False

    def _validate_single(self, epoch, model, device, vis_indices):
        labeled = correct = 0
        vis_images = []
        pbar = tqdm(range(len(self.dataset)), desc=f'Epoch {epoch} val',
                    unit='samples', ncols=100, leave=False)
        for idx in pbar:
            try:
                data = self.dataset[idx]
                img, label, modal_x, name = data['data'], data['label'], data['modal_x'], data['fn']

                pred = self.direct_eval_rgbX(img, modal_x, model, device)

                hist_tmp, labeled_tmp, correct_tmp = hist_info(self.config.num_classes, pred, label)
                self.hist += hist_tmp
                correct += correct_tmp
                labeled += labeled_tmp

                if idx in vis_indices:
                    vis_images.append((img, label, pred, name))
            except Exception as e:
                pbar.write(f"Error on validation sample {idx}: {e}")
                torch.cuda.empty_cache()
                continue
            if idx % 10 == 0:
                torch.cuda.empty_cache()
        pbar.close()
        return labeled, correct, vis_images

    def _validate_batched(self, epoch, model, device, vis_indices):
        labeled = correct = 0
        vis_images = []
        flip_tta = bool(getattr(self.config, 'eval_flip', False))
        pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader),
                    desc=f'Epoch {epoch} val', unit='batch', ncols=100, leave=False)
        for batch_idx, batch_data in pbar:
            try:
                imgs_raw = batch_data['data']
                labels = batch_data['label']
                modal_xs_raw = batch_data['modal_x']
                names = batch_data['fn']

                processed_imgs, processed_modal_xs = [], []
                for i in range(len(imgs_raw)):
                    img = self._to_hwc_numpy(imgs_raw[i])
                    modal_x = self._to_hwc_numpy(modal_xs_raw[i])
                    p_img, p_modal_x = self.process_image_rgbX(img, modal_x)
                    processed_imgs.append(p_img)
                    processed_modal_xs.append(p_modal_x)

                imgs = torch.stack([torch.from_numpy(a) for a in processed_imgs]).float().to(device)
                modal_xs = torch.stack([torch.from_numpy(a) for a in processed_modal_xs]).float().to(device)

                preds = model(imgs, modal_xs)
                if isinstance(preds, (list, tuple)):
                    preds = preds[0]
                if flip_tta:
                    preds_flip = model(imgs.flip(-1), modal_xs.flip(-1))
                    if isinstance(preds_flip, (list, tuple)):
                        preds_flip = preds_flip[0]
                    preds = (preds + preds_flip.flip(-1)) * 0.5
                preds = torch.argmax(preds, dim=1).cpu().numpy().astype(np.int32)

                for i in range(len(imgs)):
                    pred = preds[i]
                    label = labels[i].numpy().astype(np.int32)
                    name = names[i]

                    if pred.shape != label.shape:
                        logger.warning(f"Shape mismatch: pred={pred.shape}, label={label.shape}")
                        continue

                    # Out-of-range labels are treated as ignore (255).
                    label = np.where((label >= 0) & (label < self.config.num_classes), label, 255)

                    try:
                        hist_tmp, labeled_tmp, correct_tmp = hist_info(self.config.num_classes, pred, label)
                        if labeled_tmp > 0:
                            self.hist += hist_tmp
                            correct += correct_tmp
                            labeled += labeled_tmp
                    except Exception as metric_e:
                        logger.warning(f"Metric computation failed: {metric_e}")
                        continue

                    sample_idx = batch_idx * self.config.eval_batch_size + i
                    if sample_idx in vis_indices:
                        img_vis = self._to_hwc_numpy(imgs_raw[i])
                        if img_vis.dtype != np.uint8:
                            img_vis = np.clip(img_vis, 0, 255).astype(np.uint8)
                        vis_images.append((img_vis, label, pred, name))

            except Exception as e:
                pbar.write(f"Error on validation batch {batch_idx}: {e}")
                continue
        pbar.close()
        return labeled, correct, vis_images

    # ------------------------------------------------------------------

    def print_results(self, iou, mean_IoU, freq_IoU, mean_pixel_acc, pixel_acc):
        n_eval = getattr(self.config, 'num_eval_classes', None)
        if n_eval is None:
            n_eval = self.config.num_classes

        result_str = 'IoU:'
        for i in range(self.config.num_classes):
            marker = '' if i < n_eval else ' (excluded)'
            result_str += ' {}: {:.4f}{}'.format(self.config.class_names[i], iou[i], marker)
        result_str += '\nMean IoU ({}/{} classes): {:.4f}, Freq IoU: {:.4f}\n'.format(
            n_eval, self.config.num_classes, mean_IoU, freq_IoU)

        classAcc = np.diag(self.hist) / (self.hist.sum(axis=1) + 1e-10)
        result_str += 'Pixel Acc:'
        for i in range(self.config.num_classes):
            if self.hist.sum(axis=1)[i] > 0:
                result_str += ' {}: {:.4f}'.format(self.config.class_names[i], classAcc[i])
            else:
                result_str += ' {}: N/A'.format(self.config.class_names[i])
        result_str += '\nMean Pixel Acc: {:.4f}, Pixel Acc: {:.4f}\n'.format(mean_pixel_acc, pixel_acc)
        return result_str

    def save_visualizations(self, vis_images, epoch):
        save_dir = os.path.join(self.config.root_dir, 'val_results', f'epoch_{epoch}')
        ensure_dir(save_dir)
        colors = build_class_colors(self.config.num_classes,
                                    getattr(self.config, 'dataset_name', ''))

        for img, label, pred, name in vis_images:
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.imshow(colorize_mask(label.astype(np.int32), colors))
            plt.title('Ground Truth')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(colorize_mask(pred.astype(np.int32), colors))
            plt.title('Prediction')
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{name}.png'))
            plt.close()

        logger.info(f'Saved {len(vis_images)} visualizations to {save_dir}')

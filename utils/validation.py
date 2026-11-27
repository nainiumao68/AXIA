import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metric import hist_info, compute_score
from dataloader.RGBXDataset import RGBXDataset
from engine.logger import get_logger

logger = get_logger()


class ValidationHelper:
    def __init__(self, config, network, devices):
        self.config = config
        self.network = network
        self.devices = devices
        self.best_miou = 0.0
        self.best_epoch = 0
        self.rgb_norm_mean, self.rgb_norm_std = config.rgb_norm_mean, config.rgb_norm_std
        self.x_norm_mean, self.x_norm_std = config.x_norm_mean, config.x_norm_std

        data_setting = {'rgb_root': config.rgb_root_folder,
                        'rgb_format': config.rgb_format,
                        'gt_root': config.gt_root_folder,
                        'gt_format': config.gt_format,
                        'transform_gt': config.gt_transform,
                        'num_classes': config.num_classes,
                        'ignore_index': getattr(config, 'background', 255),
                        'label_mapping': getattr(config, 'label_mapping', None),
                        'x_root': config.x_root_folder,
                        'x_format': config.x_format,
                        'x_single_channel': config.x_is_single_channel,
                        'train_source': config.train_source,
                        'eval_source': config.eval_source,
                        'gt_color_palette': getattr(config, 'gt_color_palette', None)}

        self.dataset = RGBXDataset(data_setting, 'val')

        self.val_loader = None
        if (getattr(config, 'eval_batch_size', 1) > 1
                and getattr(config, 'dataset_name', '').upper() != 'CART'):
            self.val_loader = DataLoader(
                self.dataset,
                batch_size=config.eval_batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                drop_last=False
            )

    def validate(self, epoch, model):
        logger.info(f"开始验证 Epoch {epoch}...")
        model.eval()

        device = self.devices[0]

        self.hist = np.zeros((self.config.num_classes, self.config.num_classes))
        correct = 0
        labeled = 0

        try:
            with torch.no_grad():
                if self.val_loader is not None:
                    logger.info(f"使用batch验证，batch_size={self.config.eval_batch_size}")
                    pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader),
                                desc=f'Epoch {epoch} 验证', unit='batch', ncols=100, leave=False)

                    for batch_idx, batch_data in pbar:
                        try:
                            imgs_raw = batch_data['data']
                            labels = batch_data['label']
                            modal_xs_raw = batch_data['modal_x']

                            processed_imgs = []
                            processed_modal_xs = []
                            for i in range(len(imgs_raw)):
                                img = self._to_hwc_numpy(imgs_raw[i])
                                modal_x = self._to_hwc_numpy(modal_xs_raw[i])
                                img_processed, modal_x_processed = self.process_image_rgbX(img, modal_x)
                                processed_imgs.append(img_processed)
                                processed_modal_xs.append(modal_x_processed)

                            imgs = torch.stack([torch.from_numpy(img) for img in processed_imgs]).float().to(device)
                            modal_xs = torch.stack([torch.from_numpy(x) for x in processed_modal_xs]).float().to(device)

                            preds = self._predict(imgs, modal_xs, model)

                            for i in range(len(imgs)):
                                pred = preds[i]
                                label = labels[i].numpy().astype(np.int32)

                                if pred.shape != label.shape:
                                    logger.warning(f"预测和标签形状不匹配: pred={pred.shape}, label={label.shape}")
                                    continue

                                if np.any(label >= self.config.num_classes) or np.any(label < 0):
                                    label = np.where((label >= 0) & (label < self.config.num_classes), label, 255)

                                try:
                                    hist_tmp, labeled_tmp, correct_tmp = hist_info(self.config.num_classes, pred, label)
                                    if labeled_tmp > 0:
                                        self.hist += hist_tmp
                                        correct += correct_tmp
                                        labeled += labeled_tmp
                                except Exception as metric_e:
                                    logger.warning(f"计算指标时出错: {metric_e}")
                                    continue

                        except Exception as e:
                            pbar.write(f"处理验证批次 {batch_idx} 时出错: {e}")
                            continue

                    pbar.close()
                else:
                    logger.info("使用单样本验证")
                    pbar = tqdm(range(len(self.dataset)), desc=f'Epoch {epoch} 验证',
                                unit='samples', ncols=100, leave=False)

                    for idx in pbar:
                        try:
                            data = self.dataset[idx]
                            pred = self.direct_eval_rgbX(data['data'], data['modal_x'], model, device)

                            hist_tmp, labeled_tmp, correct_tmp = hist_info(
                                self.config.num_classes, pred, data['label'])
                            self.hist += hist_tmp
                            correct += correct_tmp
                            labeled += labeled_tmp

                            if idx % 10 == 0:
                                torch.cuda.empty_cache()
                        except Exception as e:
                            pbar.write(f"处理验证样本 {idx} 时出错: {e}")
                            torch.cuda.empty_cache()
                            continue

                    pbar.close()

            if labeled == 0:
                logger.warning("验证过程中没有有效的标注像素，跳过本次验证")
                return 0.0, False

            if np.sum(self.hist) == 0:
                logger.warning("混淆矩阵为空，跳过本次验证")
                return 0.0, False

            try:
                n_eval = getattr(self.config, 'num_eval_classes', None)
                iou, mean_IoU, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(
                    self.hist, correct, labeled, num_eval_classes=n_eval)

                if np.isnan(mean_IoU) or np.isinf(mean_IoU):
                    logger.warning("计算得到的mIoU无效，设为0")
                    mean_IoU = 0.0

                if np.isnan(pixel_acc) or np.isinf(pixel_acc):
                    logger.warning("计算得到的pixel_acc无效，设为0")
                    pixel_acc = 0.0

                result_line = self.print_results(iou, mean_IoU, freq_IoU, mean_pixel_acc, pixel_acc)
                logger.info(f'Validation Epoch {epoch}:\n{result_line}')

                is_best = mean_IoU > self.best_miou
                if is_best:
                    self.best_miou = mean_IoU
                    self.best_epoch = epoch
                    logger.info(f'找到新的最佳模型! mIoU: {mean_IoU:.4f}')

                return mean_IoU, is_best

            except Exception as compute_e:
                logger.error(f"计算验证指标时出错: {compute_e}")
                return 0.0, False
        except Exception as e:
            logger.error(f"验证过程中出错: {e}")
            return 0.0, False

    def print_results(self, iou, mean_IoU, freq_IoU, mean_pixel_acc, pixel_acc):
        n_eval = getattr(self.config, 'num_eval_classes', None)
        if n_eval is None:
            n_eval = self.config.num_classes
        result_str = 'IoU:'
        for i in range(self.config.num_classes):
            marker = '' if i < n_eval else ' (excluded)'
            result_str += ' {}: {:.4f}{}'.format(self.config.class_names[i], iou[i], marker)
        result_str += '\n'
        result_str += 'Mean IoU ({}/{} classes): {:.4f}, Freq IoU: {:.4f}\n'.format(
            n_eval, self.config.num_classes, mean_IoU, freq_IoU)

        classAcc = np.diag(self.hist) / (self.hist.sum(axis=1) + 1e-10)

        result_str += 'Pixel Acc:'
        for i in range(self.config.num_classes):
            if self.hist.sum(axis=1)[i] > 0:
                result_str += ' {}: {:.4f}'.format(self.config.class_names[i], classAcc[i])
            else:
                result_str += ' {}: N/A'.format(self.config.class_names[i])
        result_str += '\n'
        result_str += 'Mean Pixel Acc: {:.4f}, Pixel Acc: {:.4f}\n'.format(mean_pixel_acc, pixel_acc)

        result_str += '\n混淆矩阵对角线元素（正确预测的像素数）:\n'
        for i in range(self.config.num_classes):
            result_str += '{}: {}\n'.format(self.config.class_names[i], self.hist[i, i])

        result_str += '\n每个类别在ground truth中的像素总数:\n'
        for i in range(self.config.num_classes):
            result_str += '{}: {}\n'.format(self.config.class_names[i], self.hist.sum(axis=1)[i])

        result_str += '\n每个类别在预测结果中的像素总数:\n'
        for i in range(self.config.num_classes):
            result_str += '{}: {}\n'.format(self.config.class_names[i], self.hist.sum(axis=0)[i])

        return result_str

    @staticmethod
    def _to_hwc_numpy(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if value.ndim == 3 and value.shape[0] == 3:
            value = value.transpose(1, 2, 0)
        return value

    def _predict(self, imgs, modal_xs, model):
        preds = model(imgs, modal_xs)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if bool(getattr(self.config, 'eval_flip', False)):
            preds_flip = model(imgs.flip(-1), modal_xs.flip(-1))
            if isinstance(preds_flip, (list, tuple)):
                preds_flip = preds_flip[0]
            preds = (preds + preds_flip.flip(-1)) * 0.5
        return torch.argmax(preds, dim=1).cpu().numpy().astype(np.int32)

    def direct_eval_rgbX(self, img, modal_x, model, device):
        model.eval()
        img = self._to_hwc_numpy(img)
        modal_x = self._to_hwc_numpy(modal_x)
        h, w = img.shape[:2]
        img, modal_x = self.process_image_rgbX(img, modal_x)
        rgb = torch.from_numpy(np.ascontiguousarray(img[None])).float().to(device)
        x = torch.from_numpy(np.ascontiguousarray(modal_x[None])).float().to(device)
        with torch.no_grad():
            score = self._predict(rgb, x, model)
        if score.shape[1:] != (h, w):
            score = cv2.resize(score, (w, h), interpolation=cv2.INTER_NEAREST)
        return score[0]

    def process_image_rgbX(self, img, modal_x):
        p_img = self.normalize(img, self.rgb_norm_mean, self.rgb_norm_std)

        if len(modal_x.shape) == 2:
            p_modal_x = self.normalize(modal_x, 0, 1)[np.newaxis, ...]
        else:
            p_modal_x = self.normalize(modal_x, self.x_norm_mean, self.x_norm_std).transpose(2, 0, 1)

        return p_img.transpose(2, 0, 1), p_modal_x

    def normalize(self, img, mean, std):
        img = img.astype(np.float32) / 255.0
        mean = np.array(mean)
        std = np.array(std)

        if len(img.shape) == 3:
            mean = mean.reshape(1, 1, -1)
            std = std.reshape(1, 1, -1)

        return (img - mean) / std

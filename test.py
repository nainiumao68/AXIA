import os.path as osp
import argparse
import time
import json
from typing import List

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tabulate import tabulate
from tqdm import tqdm

from engine.logger import get_logger
from utils.pyt_utils import ensure_dir, load_model, load_config
from utils.metric import hist_info, compute_score
from utils.validation import ValidationHelper
from utils.visualize import build_class_colors, colorize_mask

logger = get_logger()


def build_model(config, device):
    from models.builder_AXIA import EncoderDecoder
    config.pretrained_model = None
    config.sar_pretrained_model = None
    criterion = nn.CrossEntropyLoss(reduction='mean',
                                    ignore_index=getattr(config, 'background', 255))
    model = EncoderDecoder(cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d)
    return model.to(device).eval()


def load_weights(model, weights_path):
    if not osp.isabs(weights_path):
        weights_path = osp.abspath(weights_path)
    if not osp.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    ckpt = torch.load(weights_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model' in ckpt:
        ckpt = ckpt['model']
    return load_model(model, ckpt)


class EvaluationCollator:
    normalize = ValidationHelper.normalize
    process_image_rgbX = ValidationHelper.process_image_rgbX

    def __init__(self, helper):
        self.rgb_norm_mean = helper.rgb_norm_mean
        self.rgb_norm_std = helper.rgb_norm_std
        self.x_norm_mean = helper.x_norm_mean
        self.x_norm_std = helper.x_norm_std

    def __call__(self, samples: List[dict]):
        imgs, xs, labels, names, sizes, orig_imgs, orig_xs = [], [], [], [], [], [], []
        for s in samples:
            img = ValidationHelper._to_hwc_numpy(s['data'])
            x = ValidationHelper._to_hwc_numpy(s['modal_x'])
            sizes.append(img.shape[:2])

            p_img, p_x = self.process_image_rgbX(img, x)
            imgs.append(torch.from_numpy(p_img).float())
            xs.append(torch.from_numpy(p_x).float())

            label = s['label']
            labels.append(torch.from_numpy(label) if isinstance(label, np.ndarray) else label)
            names.append(s['fn'])
            orig_imgs.append(img)
            orig_xs.append(x)

        return {
            'img': torch.stack(imgs, dim=0),
            'x': torch.stack(xs, dim=0),
            'label': labels,
            'name': names,
            'size': sizes,
            'orig_img': orig_imgs,
            'orig_x': orig_xs,
        }


def build_dataloader(dataset, helper, batch_size, num_workers, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=EvaluationCollator(helper),
    )


def _save_figure(path, fig_size, draw, bbox_tight=False):
    plt.figure(figsize=fig_size, dpi=100)
    draw()
    if bbox_tight:
        plt.tight_layout(pad=0)
        plt.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0)
    else:
        plt.tight_layout()
        plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()


def run_eval(config_path, weights_path, output_dir=None, device_str=None,
             batch_size=4, num_workers=4, amp=False, visualize=False,
             vis_limit=0, vis_seg_only=False, results_json=None):
    device = torch.device(
        'cpu' if device_str is None and not torch.cuda.is_available()
        else (device_str or 'cuda'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        logger.warning('CUDA unavailable, falling back to CPU')
        device = torch.device('cpu')

    torch.backends.cudnn.benchmark = True

    config = load_config(config_path)
    dataset_name = getattr(config, 'dataset_name', 'Unknown')
    num_classes = getattr(config, 'num_classes', None)
    if num_classes is None:
        raise ValueError('num_classes is not set in the config')

    if output_dir is None:
        base = osp.splitext(osp.basename(weights_path))[0]
        output_dir = osp.join(getattr(config, 'root_dir', '.'), f'val_results_{dataset_name}_{base}')
    ensure_dir(output_dir)

    model = load_weights(build_model(config, device), weights_path)

    helper = ValidationHelper(config, model, [device])
    dataset = helper.dataset
    loader = build_dataloader(dataset, helper, batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=torch.cuda.is_available())

    class_colors = build_class_colors(num_classes, dataset_name)

    hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    correct = 0
    labeled = 0
    saved_vis = 0
    saved_vis_seg = 0

    start_t = time.time()

    with torch.inference_mode():
        pbar = tqdm(total=len(dataset), desc=f'{dataset_name} Testing', ncols=100)
        for batch in loader:
            imgs = batch['img'].to(device, non_blocking=True)
            xs = batch['x'].to(device, non_blocking=True)
            labels = batch['label']
            names = batch['name']
            sizes = batch['size']

            with torch.autocast(device_type='cuda', dtype=torch.float16,
                                enabled=amp and device.type == 'cuda'):
                logits = model(imgs, xs)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)

            bsz = preds.shape[0]
            for i in range(bsz):
                h, w = sizes[i]
                name = names[i]
                pred_i = preds[i]
                if pred_i.shape[:2] != (h, w):
                    pred_i = cv2.resize(pred_i, (w, h), interpolation=cv2.INTER_NEAREST)

                label_i = labels[i].numpy() if hasattr(labels[i], 'numpy') else np.asarray(labels[i])

                hist_tmp, labeled_tmp, correct_tmp = hist_info(num_classes, pred_i, label_i)
                hist += hist_tmp
                correct += correct_tmp
                labeled += labeled_tmp

                if vis_seg_only and (vis_limit <= 0 or saved_vis_seg < vis_limit):
                    try:
                        pred_vis = colorize_mask(pred_i.astype(np.int32), class_colors)
                        vis_dir = osp.join(output_dir, 'seg_predictions')
                        ensure_dir(vis_dir)
                        _save_figure(
                            osp.join(vis_dir, f'{name}.png'),
                            (pred_vis.shape[1] / 100, pred_vis.shape[0] / 100),
                            lambda: (plt.imshow(pred_vis), plt.axis('off')),
                            bbox_tight=True)
                        saved_vis_seg += 1
                    except Exception as e:
                        plt.close('all')
                        logger.warning(f"Failed to save prediction map for {name}: {e}")

                if visualize and (vis_limit <= 0 or saved_vis < vis_limit):
                    try:
                        img_vis = batch['orig_img'][i]
                        x_vis = batch['orig_x'][i]
                        if isinstance(img_vis, np.ndarray) and img_vis.dtype != np.uint8:
                            img_vis = (np.clip(img_vis, 0, 255)).astype(np.uint8) \
                                if img_vis.max() > 1.0 else (np.clip(img_vis * 255, 0, 255)).astype(np.uint8)

                        def draw():
                            plt.subplot(1, 4, 1)
                            plt.imshow(img_vis)
                            plt.title('RGB Image', fontsize=10)
                            plt.axis('off')
                            plt.subplot(1, 4, 2)
                            plt.imshow(x_vis[:, :, 0] if isinstance(x_vis, np.ndarray) and x_vis.ndim == 3 else x_vis,
                                       cmap='gray')
                            plt.title('X Modality', fontsize=10)
                            plt.axis('off')
                            plt.subplot(1, 4, 3)
                            plt.imshow(colorize_mask(label_i.astype(np.int32), class_colors))
                            plt.title('Ground Truth', fontsize=10)
                            plt.axis('off')
                            plt.subplot(1, 4, 4)
                            plt.imshow(colorize_mask(pred_i.astype(np.int32), class_colors))
                            plt.title('Prediction', fontsize=10)
                            plt.axis('off')

                        vis_dir = osp.join(output_dir, 'visualizations')
                        ensure_dir(vis_dir)
                        _save_figure(osp.join(vis_dir, f'{name}.png'), (16, 4), draw)
                    except Exception:
                        plt.close('all')

                    saved_vis += 1

            del logits, preds
            pbar.update(bsz)

        pbar.close()

    elapsed = time.time() - start_t

    num_eval_classes = getattr(config, 'num_eval_classes', None)
    iou, mean_IoU, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(
        hist, correct, labeled, num_eval_classes=num_eval_classes)

    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    for i in range(num_classes):
        precision[i] = hist[i, i] / (hist[:, i].sum() + 1e-10)
        recall[i] = hist[i, i] / (hist[i, :].sum() + 1e-10)

    logger.info("=" * 100)
    logger.info(f"AXIA evaluation report | dataset: {dataset_name}")
    logger.info("=" * 100)
    headers = ["Class", "IoU", "Precision", "Recall", "Accuracy", "GT_Pixels", "Pred_Pixels"]
    cls_names = getattr(config, 'class_names', [str(i) for i in range(num_classes)])
    table_data = []
    for i in range(num_classes):
        table_data.append([
            cls_names[i] if i < len(cls_names) else str(i),
            f"{iou[i]:.4f}",
            f"{precision[i]:.4f}",
            f"{recall[i]:.4f}",
            f"{recall[i]:.4f}",
            f"{int(hist.sum(axis=1)[i]):,}",
            f"{int(hist.sum(axis=0)[i]):,}",
        ])
    table_data.append([
        "Mean",
        f"{mean_IoU:.4f}",
        f"{np.nanmean(precision):.4f}",
        f"{np.nanmean(recall):.4f}",
        f"{mean_pixel_acc:.4f}",
        "-",
        "-",
    ])
    logger.info(f"\n{tabulate(table_data, headers=headers, tablefmt='grid')}")
    logger.info(f"Overall: mIoU={mean_IoU:.4f}, OA={pixel_acc:.4f}, "
                f"MeanAcc={mean_pixel_acc:.4f}, FreqIoU={freq_IoU:.4f}")
    logger.info(f"Samples: {len(dataset)} | total: {elapsed:.2f}s | "
                f"avg: {elapsed / max(1, len(dataset)):.3f}s/img")
    logger.info("=" * 100)

    result = {
        'model': 'AXIA',
        'backbone': getattr(config, 'backbone', 'dual_encoder'),
        'dataset': dataset_name,
        'classes': list(cls_names),
        'acc': [None if np.isnan(v) else float(v) for v in recall],
        'iou': [None if np.isnan(v) else float(v) for v in iou],
        'macc': None if np.isnan(mean_pixel_acc) else float(mean_pixel_acc),
        'miou': None if np.isnan(mean_IoU) else float(mean_IoU),
        'pixel_acc': None if np.isnan(pixel_acc) else float(pixel_acc),
        'num_eval_classes': num_eval_classes,
        'config': config_path,
        'checkpoint': weights_path,
        'output_dir': output_dir,
    }
    if results_json:
        ensure_dir(osp.dirname(osp.abspath(results_json)))
        with open(results_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def parse_args():
    p = argparse.ArgumentParser(description='AXIA evaluation (dataset selected by config)')
    p.add_argument('--config', type=str, required=True, help='config file, e.g. config_pie.py')
    p.add_argument('--weights', type=str, required=True, help='checkpoint, e.g. best_model.pth')
    p.add_argument('-o', '--output', type=str, default=None,
                   help='output directory (default: derived from dataset and weights name)')
    p.add_argument('--device', type=str, default=None, help='cuda / cuda:0 / cpu (default: auto)')
    p.add_argument('--batch-size', type=int, default=4, help='evaluation batch size')
    p.add_argument('--workers', type=int, default=4, help='DataLoader worker processes')
    p.add_argument('--amp', action='store_true', help='enable AMP mixed precision (default off)')
    p.add_argument('--vis', action='store_true',
                   help='save 4-panel visualizations (RGB / X / GT / Pred)')
    p.add_argument('--vis-limit', type=int, default=0,
                   help='max number of visualizations to save (0 = no limit)')
    p.add_argument('--vis-seg-only', action='store_true',
                   help='save only the colorized prediction maps')
    p.add_argument('--results-json', type=str, default=None,
                   help='save metrics as JSON to this path')
    return p.parse_args()


def main():
    args = parse_args()
    run_eval(
        config_path=args.config,
        weights_path=args.weights,
        output_dir=args.output,
        device_str=args.device,
        batch_size=max(1, int(args.batch_size)),
        num_workers=max(0, int(args.workers)),
        amp=args.amp,
        visualize=args.vis,
        vis_limit=int(args.vis_limit),
        vis_seg_only=args.vis_seg_only,
        results_json=args.results_json,
    )



if __name__ == '__main__':
    main()

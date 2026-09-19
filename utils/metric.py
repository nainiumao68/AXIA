"""Confusion-matrix based segmentation metrics (mIoU, pixel accuracy, ...)."""
import numpy as np

np.seterr(divide='ignore', invalid='ignore')


def hist_info(n_cl, pred, gt):
    """Accumulate the confusion matrix for one prediction/label pair.

    Returns ``(hist, labeled, correct)`` where pixels with labels outside
    ``[0, n_cl)`` or equal to 255 are ignored.
    """
    if hasattr(pred, 'cpu'):
        pred = pred.cpu().numpy()
    if hasattr(gt, 'cpu'):
        gt = gt.cpu().numpy()

    assert pred.shape == gt.shape, f"Shape mismatch: pred={pred.shape}, gt={gt.shape}"

    pred = pred.astype(np.int32)
    gt = gt.astype(np.int32)

    k = (gt >= 0) & (gt < n_cl) & (gt != 255)
    labeled = np.sum(k)

    if labeled == 0:
        return np.zeros((n_cl, n_cl), dtype=np.int64), 0, 0

    correct = np.sum((pred[k] == gt[k]))

    pred_valid = np.clip(pred[k], 0, n_cl - 1)
    gt_valid = gt[k]

    confusionMatrix = np.bincount(n_cl * gt_valid.astype(int) + pred_valid.astype(int),
                                  minlength=n_cl ** 2).reshape(n_cl, n_cl)
    return confusionMatrix, labeled, correct


def compute_score(hist, correct, labeled, num_eval_classes=None):
    """Compute mIoU and related metrics from the confusion matrix.

    Args:
        num_eval_classes: If set, only the first *num_eval_classes* classes
            participate in mIoU / mean-acc averaging (ISPRS convention:
            set to 5 to exclude the clutter class).
    """
    eps = 1e-10

    denominator = hist.sum(1) + hist.sum(0) - np.diag(hist) + eps
    iou = np.diag(hist) / denominator

    n_cls = len(iou)
    if num_eval_classes is None:
        num_eval_classes = n_cls

    eval_iou = iou[:num_eval_classes]
    valid_classes = (hist.sum(1) + hist.sum(0) - np.diag(hist))[:num_eval_classes] > eps
    if np.any(valid_classes):
        mean_IoU = np.mean(eval_iou[valid_classes])
    else:
        mean_IoU = 0.0
    mean_IoU_no_back = mean_IoU

    hist_sum = hist.sum()
    if hist_sum > 0:
        freq = hist.sum(1)[:num_eval_classes] / hist_sum
        valid_freq = freq > eps
        if np.any(valid_freq):
            freq_IoU = (eval_iou[valid_freq] * freq[valid_freq]).sum()
        else:
            freq_IoU = 0.0
    else:
        freq_IoU = 0.0

    class_totals = hist.sum(axis=1) + eps
    classAcc = np.diag(hist) / class_totals

    valid_class_acc = hist.sum(axis=1)[:num_eval_classes] > eps
    if np.any(valid_class_acc):
        mean_pixel_acc = np.mean(classAcc[:num_eval_classes][valid_class_acc])
    else:
        mean_pixel_acc = 0.0

    if labeled > 0:
        pixel_acc = correct / labeled
    else:
        pixel_acc = 0.0

    return iou, mean_IoU, mean_IoU_no_back, freq_IoU, mean_pixel_acc, pixel_acc

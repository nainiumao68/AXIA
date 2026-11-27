import numpy as np


def hist_info(n_cl, pred, gt):
    if hasattr(pred, 'cpu'):
        pred = pred.cpu().numpy()
    if hasattr(gt, 'cpu'):
        gt = gt.cpu().numpy()

    assert pred.shape == gt.shape, f"预测和标签形状不匹配: pred={pred.shape}, gt={gt.shape}"

    pred = pred.astype(np.int32)
    gt = gt.astype(np.int32)

    k = (gt >= 0) & (gt < n_cl) & (gt != 255)
    labeled = np.sum(k)

    if labeled == 0:
        return np.zeros((n_cl, n_cl), dtype=np.int64), 0, 0

    correct = np.sum(pred[k] == gt[k])

    pred_valid = np.clip(pred[k], 0, n_cl - 1)

    confusionMatrix = np.bincount(n_cl * gt[k].astype(int) + pred_valid.astype(int),
                                  minlength=n_cl ** 2).reshape(n_cl, n_cl)
    return confusionMatrix, labeled, correct


def compute_score(hist, correct, labeled, num_eval_classes=None):
    eps = 1e-10

    denominator = hist.sum(1) + hist.sum(0) - np.diag(hist) + eps
    iou = np.diag(hist) / denominator

    if num_eval_classes is None:
        num_eval_classes = len(iou)

    eval_iou = iou[:num_eval_classes]
    valid_classes = (hist.sum(1) + hist.sum(0) - np.diag(hist))[:num_eval_classes] > eps
    mean_IoU = np.mean(eval_iou[valid_classes]) if np.any(valid_classes) else 0.0

    hist_sum = hist.sum()
    freq_IoU = 0.0
    if hist_sum > 0:
        freq = hist.sum(1)[:num_eval_classes] / hist_sum
        valid_freq = freq > eps
        if np.any(valid_freq):
            freq_IoU = (eval_iou[valid_freq] * freq[valid_freq]).sum()

    classAcc = np.diag(hist) / (hist.sum(axis=1) + eps)
    valid_class_acc = hist.sum(axis=1)[:num_eval_classes] > eps
    mean_pixel_acc = (np.mean(classAcc[:num_eval_classes][valid_class_acc])
                      if np.any(valid_class_acc) else 0.0)

    pixel_acc = correct / labeled if labeled > 0 else 0.0

    return iou, mean_IoU, freq_IoU, mean_pixel_acc, pixel_acc

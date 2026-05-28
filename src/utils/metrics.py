import numpy as np


def center_error(pred_bbox, gt_bbox):
    pred_cx = pred_bbox[0] + pred_bbox[2] / 2
    pred_cy = pred_bbox[1] + pred_bbox[3] / 2
    gt_cx = gt_bbox[0] + gt_bbox[2] / 2
    gt_cy = gt_bbox[1] + gt_bbox[3] / 2
    return np.sqrt((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2)


def overlap(pred_bbox, gt_bbox):
    x1 = max(pred_bbox[0], gt_bbox[0])
    y1 = max(pred_bbox[1], gt_bbox[1])
    x2 = min(pred_bbox[0] + pred_bbox[2], gt_bbox[0] + gt_bbox[2])
    y2 = min(pred_bbox[1] + pred_bbox[3], gt_bbox[1] + gt_bbox[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_pred = pred_bbox[2] * pred_bbox[3]
    area_gt = gt_bbox[2] * gt_bbox[3]
    return inter / (area_pred + area_gt - inter + 1e-8)


def precision_score(errors, threshold=20):
    return np.mean(errors <= threshold)


def success_score(overlaps, threshold=0.5):
    return np.mean(overlaps > threshold)


def compute_precision_curve(errors, max_thresh=50):
    thresholds = np.linspace(0, max_thresh, 51)
    precisions = [np.mean(errors <= t) for t in thresholds]
    return thresholds, np.array(precisions)


def compute_success_curve(overlaps):
    thresholds = np.linspace(0, 1, 21)
    successes = [np.mean(overlaps >= t) for t in thresholds]
    return thresholds, np.array(successes)


def auc_score(overlaps):
    thresholds = np.linspace(0, 1, 21)
    return np.trapz([np.mean(overlaps >= t) for t in thresholds], thresholds)

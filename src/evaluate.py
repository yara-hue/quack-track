import os
import cv2
import numpy as np
import torch
from tqdm import tqdm

from .tracker import SiamRPNTracker
from .data.dataset import _read_annotation, _list_sequences
from .utils.metrics import center_error, overlap, compute_precision_curve, compute_success_curve


def evaluate_uav20l(model, data_root, device='cuda', template_sz=127, search_sz=255):
    tracker = SiamRPNTracker(model, device, template_sz, search_sz)
    seq_names = _list_sequences(data_root)

    all_errors = []
    all_overlaps = []
    seq_results = {}

    for seq_name in tqdm(seq_names, desc='Evaluating'):
        img_dir = os.path.join(data_root, 'data_seq', seq_name, 'img')
        anno_path = os.path.join(data_root, 'anno', f'{seq_name}.txt')
        if not os.path.isdir(img_dir) or not os.path.isfile(anno_path):
            continue

        imgs = sorted(os.listdir(img_dir))
        gt_bboxes = _read_annotation(anno_path)
        n = min(len(imgs), len(gt_bboxes))

        if n < 2:
            continue

        first_img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, imgs[0])), cv2.COLOR_BGR2RGB)
        tracker.init(first_img, gt_bboxes[0])

        errors = []
        overlaps = []

        for i in range(1, n):
            img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, imgs[i])), cv2.COLOR_BGR2RGB)
            pred = tracker.track(img)
            gt = gt_bboxes[i]

            err = center_error(pred, gt)
            ov = overlap(pred, gt)

            errors.append(err)
            overlaps.append(ov)

        seq_results[seq_name] = {
            'errors': errors,
            'overlaps': overlaps,
            'mean_error': np.mean(errors),
            'mean_overlap': np.mean(overlaps)
        }
        all_errors.extend(errors)
        all_overlaps.extend(overlaps)

    all_errors = np.array(all_errors)
    all_overlaps = np.array(all_overlaps)

    precision_thresh, precision_curve = compute_precision_curve(all_errors)
    success_thresh, success_curve = compute_success_curve(all_overlaps)

    results = {
        'mean_precision': np.mean(all_errors <= 20),
        'mean_success': np.mean(all_overlaps >= 0.5),
        'auc': np.trapz(success_curve, success_thresh),
        'precision_curve': (precision_thresh, precision_curve),
        'success_curve': (success_thresh, success_curve),
        'seq_results': seq_results,
        'all_errors': all_errors,
        'all_overlaps': all_overlaps,
    }

    return results

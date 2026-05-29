import os
import cv2
import numpy as np
import torch
from tqdm import tqdm

from .tracker import SiamRPNTracker
from .data.dataset import _read_annotation, _seq_img_dir
from .utils.metrics import center_error, overlap, compute_precision_curve, compute_success_curve


def _find_annotation_file(anno_dir, seq_name):
    candidates = [
        os.path.join(anno_dir, f'{seq_name}.txt'),
        os.path.join(anno_dir, 'UAV20L', f'{seq_name}.txt'),
        os.path.join(anno_dir, 'att', f'{seq_name}.txt'),
        os.path.join(anno_dir, 'UAV20L', 'att', f'{seq_name}.txt'),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    return None


def _list_uav20l_sequences(data_root):
    ext_dirs = ['', 'UAV20L']
    seqs = set()
    for sub in ext_dirs:
        base = os.path.join(data_root, sub)
        for f in sorted(os.listdir(base)):
            if f.endswith('.txt'):
                seqs.add(f.replace('.txt', ''))
    return sorted(seqs)


def evaluate_uav20l(model, data_root, img_root=None, device='cuda', template_sz=127, search_sz=255):
    tracker = SiamRPNTracker(model, device, template_sz, search_sz)
    seq_names = _list_uav20l_sequences(data_root)
    if len(seq_names) == 0:
        return {'mean_precision': 0, 'mean_success': 0, 'auc': 0, 'seq_results': {},
                'all_errors': np.array([]), 'all_overlaps': np.array([])}

    all_errors = []
    all_overlaps = []
    seq_results = {}

    for seq_name in tqdm(seq_names, desc='Evaluating'):
        if img_root:
            img_dir = _seq_img_dir(os.path.join(img_root, 'data_seq'), seq_name)
            if img_dir is None:
                img_dir = _seq_img_dir(os.path.join(img_root, 'data_seq', 'UAV123'), seq_name)
        else:
            img_dir = _seq_img_dir(os.path.join(data_root, 'data_seq'), seq_name)
        if img_dir is None:
            continue

        anno_path = _find_annotation_file(data_root, seq_name)
        if anno_path is None:
            continue

        imgs = sorted(os.listdir(img_dir))
        gt_bboxes = _read_annotation(anno_path)
        n = min(len(imgs), len(gt_bboxes))

        if n < 2:
            continue

        first_img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, imgs[0])), cv2.COLOR_BGR2RGB)
        if gt_bboxes[0] is None:
            continue
        tracker.init(first_img, gt_bboxes[0])

        errors = []
        overlaps = []

        for i in range(1, n):
            img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, imgs[i])), cv2.COLOR_BGR2RGB)
            pred = tracker.track(img)
            gt = gt_bboxes[i]
            if gt is None:
                continue

            err = center_error(pred, gt)
            ov = overlap(pred, gt)

            errors.append(err)
            overlaps.append(ov)

        if len(errors) == 0:
            continue

        seq_results[seq_name] = {
            'errors': errors,
            'overlaps': overlaps,
            'mean_error': float(np.mean(errors)),
            'mean_overlap': float(np.mean(overlaps))
        }
        all_errors.extend(errors)
        all_overlaps.extend(overlaps)

    all_errors = np.array(all_errors)
    all_overlaps = np.array(all_overlaps)

    results = {
        'mean_precision': float(np.mean(all_errors <= 20)) if len(all_errors) > 0 else 0.0,
        'mean_success': float(np.mean(all_overlaps >= 0.5)) if len(all_overlaps) > 0 else 0.0,
        'auc': float(np.trapz(*compute_success_curve(all_overlaps))) if len(all_overlaps) > 0 else 0.0,
        'seq_results': seq_results,
    }

    return results

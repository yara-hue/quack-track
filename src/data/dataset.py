import os
import json
import random
import cv2
import numpy as np
from torch.utils.data import Dataset


ANCHOR_RATIOS = [0.33, 0.5, 1.0, 2.0, 3.0]
STRIDE = 16


def _read_annotation(anno_path):
    bboxes = []
    with open(anno_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 4:
                x, y, w, h = map(float, parts)
                if not any(np.isnan(v) for v in [x, y, w, h]):
                    bboxes.append([x, y, w, h])
                else:
                    bboxes.append(None)
    return bboxes


def _seq_img_dir(seq_dir, name):
    candidates = [
        os.path.join(seq_dir, name, 'img'),
        os.path.join(seq_dir, name),
    ]
    for cand in candidates:
        if os.path.isdir(cand) and any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(cand)[:1]):
            return cand
    return None


def _list_sequences(data_root):
    seq_dir = os.path.join(data_root, 'data_seq')
    sequences = []
    if not os.path.isdir(seq_dir):
        return sequences

    for name in sorted(os.listdir(seq_dir)):
        if _seq_img_dir(seq_dir, name) is not None:
            sequences.append(name)
            continue
        for sub in sorted(os.listdir(os.path.join(seq_dir, name))):
            if _seq_img_dir(os.path.join(seq_dir, name), sub) is not None:
                sequences.append(sub)

    return sequences


def _gaussian_label(map_size, target_pos, sigma):
    h, w = map_size
    cx, cy = target_pos
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    dist = (xx - cx) ** 2 + (yy - cy) ** 2
    return np.exp(-0.5 * dist / sigma ** 2)


def _anchors(stride, ratios, response_sz):
    anchors = []
    for i in range(response_sz):
        for j in range(response_sz):
            cx = j * stride
            cy = i * stride
            for r in ratios:
                w = stride * np.sqrt(r)
                h = stride / np.sqrt(r)
                anchors.append([cx - w / 2, cy - h / 2, w, h])
    return np.array(anchors, dtype=np.float32)


def _bbox_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = bbox1[2] * bbox1[3]
    area2 = bbox2[2] * bbox2[3]
    return inter / (area1 + area2 - inter + 1e-8)


class UAVTrackingDataset(Dataset):
    def __init__(self, data_root, split='train', pairs_per_seq=20, template_sz=127, search_sz=255):
        self.data_root = data_root
        self.template_sz = template_sz
        self.search_sz = search_sz
        self.pairs_per_seq = pairs_per_seq

        seq_names = _list_sequences(data_root)
        split_path = os.path.join(data_root, f'{split}.json')
        if os.path.exists(split_path):
            with open(split_path) as f:
                split_seqs = json.load(f)
            seq_names = [s for s in seq_names if s in split_seqs]

        self.pairs = []
        for seq_name in seq_names:
            img_dir = os.path.join(data_root, 'data_seq', seq_name, 'img')
            anno_path = os.path.join(data_root, 'anno', f'{seq_name}.txt')
            if not os.path.isdir(img_dir) or not os.path.isfile(anno_path):
                continue
            imgs = sorted(os.listdir(img_dir))
            bboxes = _read_annotation(anno_path)
            valid_idx = [i for i, b in enumerate(bboxes) if b is not None]
            if len(valid_idx) < 2:
                continue
            for _ in range(pairs_per_seq):
                ti = random.randint(0, len(valid_idx) - 2)
                t = valid_idx[ti]
                max_gap = min(5, len(valid_idx) - 1 - ti)
                si = ti + random.randint(1, max_gap)
                s = valid_idx[si]
                self.pairs.append((seq_name, t, s, bboxes[t], bboxes[s]))

    def __len__(self):
        return len(self.pairs)

    def _get_img_dir(self):
        seq_dir = os.path.join(self.data_root, 'data_seq')
        for name in set(p[0] for p in self.pairs):
            img_dir = _seq_img_dir(seq_dir, name)
            if img_dir:
                return img_dir, name
        return None, None

    def __getitem__(self, idx):
        seq_name, t_idx, s_idx, template_bbox, search_bbox = self.pairs[idx]
        seq_dir = os.path.join(self.data_root, 'data_seq')
        img_dir = _seq_img_dir(seq_dir, seq_name)
        imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        template_img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, imgs[t_idx])), cv2.COLOR_BGR2RGB)
        search_img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, imgs[s_idx])), cv2.COLOR_BGR2RGB)
        return template_img, search_img, template_bbox, search_bbox

    def _compute_targets(self, search_bbox, crop_bbox):
        x, y, w, h = search_bbox
        cx, cy = x + w / 2, y + h / 2
        crop_x, crop_y, crop_w, crop_h = crop_bbox
        scale = self.search_sz / max(crop_w, crop_h)
        cx_crop = (cx - crop_x) * scale
        cy_crop = (cy - crop_y) * scale

        response_sz = 9
        sigma = STRIDE // 2

        gaussian = _gaussian_label((response_sz, response_sz),
                                    (cx_crop / STRIDE, cy_crop / STRIDE), sigma)
        gaussian = np.clip(gaussian, 0, 1)

        cls_target = np.stack([gaussian, 1.0 - gaussian], axis=0)
        cls_target = np.repeat(cls_target[np.newaxis, :, :, :], 5, axis=0)

        anc = _anchors(STRIDE, ANCHOR_RATIOS, response_sz)
        anc = anc.reshape(response_sz, response_sz, 5, 4)

        iou_map = np.zeros((response_sz, response_sz, 5), dtype=np.float32)
        for i in range(response_sz):
            for j in range(response_sz):
                for k in range(5):
                    iou_map[i, j, k] = _bbox_iou(anc[i, j, k], [cx - w / 2, cy - h / 2, w, h])

        pos_mask = (iou_map > 0.6).astype(np.float32)
        reg_target = np.zeros((4, response_sz, response_sz, 5), dtype=np.float32)
        for i in range(response_sz):
            for j in range(response_sz):
                for k in range(5):
                    anc_box = anc[i, j, k]
                    anc_w = anc_box[2]
                    anc_h = anc_box[3]
                    if anc_w > 0 and anc_h > 0 and iou_map[i, j, k] > 0.6:
                        reg_target[0, i, j, k] = ((cx - w / 2) - anc_box[0]) / anc_w
                        reg_target[1, i, j, k] = ((cy - h / 2) - anc_box[1]) / anc_h
                        reg_target[2, i, j, k] = np.log(w / anc_w)
                        reg_target[3, i, j, k] = np.log(h / anc_h)

        reg_target = reg_target.reshape(4, -1)
        pos_mask = pos_mask.reshape(-1)

        return cls_target, reg_target, pos_mask

class COCONegativeDataset(Dataset):
    def __init__(self, coco_root, num_samples=50000, template_sz=127, search_sz=255):
        self.coco_root = coco_root
        self.template_sz = template_sz
        self.search_sz = search_sz
        img_dir = os.path.join(coco_root, 'train2017')
        self.images = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(self.images)
        self.images = self.images[:num_samples]

    def __len__(self):
        return len(self.images) // 2

    def __getitem__(self, idx):
        img_a = cv2.cvtColor(cv2.imread(self.images[idx * 2]), cv2.COLOR_BGR2RGB)
        img_b = cv2.cvtColor(cv2.imread(self.images[idx * 2 + 1]), cv2.COLOR_BGR2RGB)
        return img_a, img_b

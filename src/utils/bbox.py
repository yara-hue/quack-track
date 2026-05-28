import numpy as np


def crop_context(img, bbox, context_factor=2.0, output_size=127):
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    s = max(w, h) * context_factor
    half = s / 2.0

    x1 = int(cx - half)
    y1 = int(cy - half)
    x2 = int(cx + half)
    y2 = int(cy + half)

    import cv2
    img_h, img_w = img.shape[:2]
    pad_l = max(0, -x1)
    pad_t = max(0, -y1)
    pad_r = max(0, x2 - img_w)
    pad_b = max(0, y2 - img_h)

    if any([pad_l, pad_t, pad_r, pad_b]):
        img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r,
                                 cv2.BORDER_CONSTANT, value=(124, 116, 104))
        x1 += pad_l
        y1 += pad_t

    crop = img[y1:y2, x1:x2]
    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)


def clip_bbox(bbox, img_shape):
    x, y, w, h = bbox
    x = max(0, min(x, img_shape[1] - 1))
    y = max(0, min(y, img_shape[0] - 1))
    w = max(1, min(w, img_shape[1] - x))
    h = max(1, min(h, img_shape[0] - y))
    return np.array([x, y, w, h], dtype=np.float32)


def bbox_from_delta(anchor, delta):
    dx, dy, dw, dh = delta
    ax, ay, aw, ah = anchor
    cx = ax + dx * aw
    cy = ay + dy * ah
    w = aw * np.exp(dw)
    h = ah * np.exp(dh)
    return np.array([cx - w / 2, cy - h / 2, w, h], dtype=np.float32)


def center_to_corner(bbox):
    x, y, w, h = bbox
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def corner_to_center(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)

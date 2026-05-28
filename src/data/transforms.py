import random
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T


class TrainTransforms:
    def __init__(self, template_size=127, search_size=255):
        self.template_size = template_size
        self.search_size = search_size
        self.color_jitter = T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def __call__(self, template_img, search_img, template_bbox, search_bbox):
        template_crop = self._crop_context(template_img, template_bbox, self.template_size)
        search_crop = self._crop_context(search_img, template_bbox, self.search_size)

        if random.random() < 0.5:
            seed = random.randint(0, 2**16)
            random.seed(seed)
            template_crop = np.array(self.color_jitter(Image.fromarray(template_crop)))
            random.seed(seed)
            search_crop = np.array(self.color_jitter(Image.fromarray(search_crop)))

        if random.random() < 0.2:
            k = random.choice([3, 5])
            template_crop = cv2.GaussianBlur(template_crop, (k, k), 0)
        if random.random() < 0.2:
            k = random.choice([3, 5])
            search_crop = cv2.GaussianBlur(search_crop, (k, k), 0)

        if random.random() < 0.2:
            template_crop = self._random_stretch(template_crop, self.template_size)
        if random.random() < 0.2:
            search_crop = self._random_stretch(search_crop, self.search_size)

        template_tensor = self._to_tensor(template_crop)
        search_tensor = self._to_tensor(search_crop)

        return template_tensor, search_tensor

    def _random_stretch(self, img, output_size):
        h, w = img.shape[:2]
        sx = 1.0 + random.uniform(-0.15, 0.15)
        sy = 1.0 + random.uniform(-0.15, 0.15)
        new_w = max(1, int(w * sx))
        new_h = max(1, int(h * sy))
        stretched = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(stretched, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

    def _crop_context(self, img, bbox, output_size):
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        context = max(w, h) * 2.0
        half = context / 2.0

        x1 = int(cx - half)
        y1 = int(cy - half)
        x2 = int(cx + half)
        y2 = int(cy + half)

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

    def _to_tensor(self, img):
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        return np.transpose(img, (2, 0, 1)).astype(np.float32)


class TestTransforms:
    def __init__(self, template_size=127, search_size=255):
        self.template_size = template_size
        self.search_size = search_size
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def __call__(self, template_img, search_img, template_bbox, search_bbox):
        return TrainTransforms.__call__(self, template_img, search_img, template_bbox, search_bbox)

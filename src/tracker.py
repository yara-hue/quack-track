import numpy as np
import torch
import torch.nn.functional as F

from .utils.bbox import clip_bbox, bbox_from_delta
from .data.dataset import ANCHOR_RATIOS, ANCHOR_SCALE, STRIDE


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class SiamRPNTracker:
    def __init__(self, model, device='cuda', template_sz=127, search_sz=255):
        self.model = model
        self.model.eval()
        self.device = device
        self.template_sz = template_sz
        self.search_sz = search_sz
        self.stride = STRIDE
        self.ratios = ANCHOR_RATIOS
        self.response_sz = (search_sz - template_sz) // self.stride + 1
        self.context_factor = 2.0
        self.window_influence = 0.4
        self._build_anchors()
        self._build_cosine_window()

    def _build_anchors(self):
        anchors = []
        for i in range(self.response_sz):
            for j in range(self.response_sz):
                cx = j * self.stride
                cy = i * self.stride
                for r in self.ratios:
                    w = self.stride * ANCHOR_SCALE * np.sqrt(r)
                    h = self.stride * ANCHOR_SCALE / np.sqrt(r)
                    anchors.append([cx - w/2, cy - h/2, w, h])
        self.anchors = np.array(anchors, dtype=np.float32)

    def _build_cosine_window(self):
        hann = np.hanning(self.response_sz)
        window = np.outer(hann, hann)
        self.cos_window = window.ravel()

    def _preprocess(self, img, bbox, output_sz):
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        s = max(w, h) * self.context_factor
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

        self._crop_origin = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
        crop = img[y1:y2, x1:x2]
        crop = cv2.resize(crop, (output_sz, output_sz), interpolation=cv2.INTER_LINEAR)
        return crop

    def _to_tensor(self, img):
        t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        return (t - torch.from_numpy(MEAN)) / torch.from_numpy(STD)

    def init(self, img, bbox):
        self.target_bbox = np.array(bbox, dtype=np.float32)
        self.img_shape = img.shape[:2]

        template = self._preprocess(img, bbox, self.template_sz)
        template_t = self._to_tensor(template).unsqueeze(0).to(self.device)

        with torch.no_grad():
            t_c3, t_c4, t_c5 = self.model.backbone(template_t)
            _, t_p4, _ = self.model.fpn(t_c3, t_c4, t_c5)
            self.template_feat = t_p4

    def track(self, img):
        search = self._preprocess(img, self.target_bbox, self.search_sz)
        search_t = self._to_tensor(search).unsqueeze(0).to(self.device)

        with torch.no_grad():
            s_c3, s_c4, s_c5 = self.model.backbone(search_t)
            _, s_p4, _ = self.model.fpn(s_c3, s_c4, s_c5)

            corr = self.model.corr(self.template_feat, s_p4)
            cls_logits, reg_deltas = self.model.rpn_head(corr)

        cls_score = torch.softmax(cls_logits, dim=1)[0, 1].cpu().numpy().ravel()
        cls_score = (1 - self.window_influence) * cls_score + self.window_influence * self.cos_window

        best_idx = int(np.argmax(cls_score))

        anchor_k = best_idx % 5
        spatial_idx = best_idx // 5
        h = spatial_idx // self.response_sz
        w = spatial_idx % self.response_sz

        delta = reg_deltas[0, anchor_k*4:(anchor_k+1)*4, h, w].cpu().numpy().tolist()

        pred_in_crop = bbox_from_delta(self.anchors[best_idx], delta)

        crop_w = self._crop_origin[2]
        crop_h = self._crop_origin[3]
        scale_x = crop_w / self.search_sz
        scale_y = crop_h / self.search_sz
        pred_in_img = np.array([
            self._crop_origin[0] + pred_in_crop[0] * scale_x,
            self._crop_origin[1] + pred_in_crop[1] * scale_y,
            pred_in_crop[2] * scale_x,
            pred_in_crop[3] * scale_y,
        ], dtype=np.float32)

        pred_in_img = clip_bbox(pred_in_img, img.shape[:2])
        self.target_bbox = pred_in_img
        return pred_in_img

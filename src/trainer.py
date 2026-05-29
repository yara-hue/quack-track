import os
import math
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from .models.loss import SiamRPNLoss
from .data.dataset import UAVTrackingDataset, COCONegativeDataset, UAV20L_SEQS, _gaussian_label, STRIDE, ANCHOR_RATIOS, ANCHOR_SCALE, _anchors, _bbox_iou
from .data.transforms import TrainTransforms


logger = logging.getLogger('quack-track')


def compute_targets(search_bbox, crop_bbox, search_sz=255, stride=16, response_sz=9):
    x, y, w, h = search_bbox
    cx, cy = x + w / 2, y + h / 2
    crop_x, crop_y, crop_w, crop_h = crop_bbox
    crop_cx = crop_x + crop_w / 2
    crop_cy = crop_y + crop_h / 2
    half = max(crop_w, crop_h)
    scale = search_sz / (2 * half)
    cx_crop = (cx - crop_cx + half) * scale
    cy_crop = (cy - crop_cy + half) * scale

    sigma = stride // 2
    gaussian = _gaussian_label((response_sz, response_sz),
                                (cx_crop / stride, cy_crop / stride), sigma)
    gaussian = torch.clamp(torch.from_numpy(gaussian).float(), 0, 1)

    cls_target = torch.stack([gaussian, 1.0 - gaussian], dim=0)
    cls_target = cls_target.unsqueeze(0).expand(5, -1, -1, -1)

    anc = _anchors(stride, ANCHOR_RATIOS, response_sz, ANCHOR_SCALE)
    anc = anc.reshape(response_sz, response_sz, 5, 4)

    num_anc = response_sz * response_sz * 5
    reg_target = torch.zeros(4, num_anc, dtype=torch.float32)
    reg_mask = torch.zeros(num_anc, dtype=torch.float32)

    sw = w * scale
    sh = h * scale
    gx, gy, gw, gh = cx_crop - sw/2, cy_crop - sh/2, sw, sh

    for i in range(response_sz):
        for j in range(response_sz):
            for k in range(5):
                anc_box = anc[i, j, k]
                anc_w = float(anc_box[2])
                anc_h = float(anc_box[3])
                if anc_w <= 0 or anc_h <= 0:
                    continue
                iou = _bbox_iou(anc_box, [gx, gy, gw, gh])
                idx = (i * response_sz + j) * 5 + k
                if iou > 0.6:
                    reg_target[0, idx] = float((gx - anc_box[0]) / anc_w)
                    reg_target[1, idx] = float((gy - anc_box[1]) / anc_h)
                    reg_target[2, idx] = math.log(gw / anc_w)
                    reg_target[3, idx] = math.log(gh / anc_h)
                    reg_mask[idx] = 1.0

    return cls_target, reg_target, reg_mask


def collate_tracking(batch):
    templates = []
    searches = []
    cls_targets = []
    reg_targets = []
    reg_masks = []

    for item in batch:
        if len(item) == 2:
            template_img, search_img = item
            t, s = TrainTransforms()(template_img, search_img, [0, 0, 127, 127], [0, 0, 255, 255])
            templates.append(torch.from_numpy(t))
            searches.append(torch.from_numpy(s))
            cls_targets.append(torch.zeros(5, 2, 9, 9, dtype=torch.float32))
            reg_targets.append(torch.zeros(4, 405, dtype=torch.float32))
            reg_masks.append(torch.zeros(405, dtype=torch.float32))
        else:
            template_img, search_img, template_bbox, search_bbox = item
            t, s = TrainTransforms()(template_img, search_img, template_bbox, search_bbox)
            templates.append(torch.from_numpy(t))
            searches.append(torch.from_numpy(s))
            cls_t, reg_t, mask = compute_targets(search_bbox, template_bbox)
            cls_targets.append(cls_t)
            reg_targets.append(reg_t)
            reg_masks.append(mask)

    return {
        'template': torch.stack(templates),
        'search': torch.stack(searches),
        'cls_target': torch.stack(cls_targets),
        'reg_target': torch.stack(reg_targets),
        'reg_mask': torch.stack(reg_masks),
    }


class Trainer:
    def __init__(self, model, cfg, device='cuda'):
        self.model = model
        self.cfg = cfg
        self.device = device

        self.criterion = SiamRPNLoss(
            cls_weight=cfg['loss']['cls_weight'],
            reg_weight=cfg['loss']['reg_weight']
        )

        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg['train']['lr'],
            momentum=cfg['train']['momentum'],
            weight_decay=cfg['train']['weight_decay']
        )

        total_epochs = cfg['train']['epochs']
        warmup_epochs = cfg['train']['warmup_epochs']
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_epochs - warmup_epochs
        )
        self.warmup_epochs = warmup_epochs

        self.start_epoch = 0
        self.best_loss = float('inf')

    def _build_dataloaders(self):
        cfg = self.cfg
        exclude = None
        if cfg['data'].get('uav20l_exclude', True):
            exclude = UAV20L_SEQS
        tracking = UAVTrackingDataset(
            data_root=cfg['data']['uav123_root'],
            pairs_per_seq=cfg['data']['pairs_per_seq'],
            exclude_seqs=exclude
        )
        if cfg['data'].get('coco_root'):
            coco = COCONegativeDataset(
                coco_root=cfg['data']['coco_root'],
                num_samples=cfg['data'].get('coco_samples', 50000)
            )
            dataset = ConcatDataset([tracking, coco])
        else:
            dataset = tracking

        return DataLoader(
            dataset,
            batch_size=cfg['train']['batch_size'],
            shuffle=True,
            num_workers=cfg['train']['num_workers'],
            collate_fn=collate_tracking,
            drop_last=True
        )

    def train(self, checkpoint_dir='checkpoints'):
        os.makedirs(checkpoint_dir, exist_ok=True)

        loader = self._build_dataloaders()
        self.model.to(self.device)
        self.model.train()

        for epoch in range(self.start_epoch, self.cfg['train']['epochs']):
            if epoch < self.warmup_epochs:
                lr = self.cfg['train']['lr'] * (epoch + 1) / self.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg['lr'] = lr

            epoch_loss = 0.0
            epoch_cls = 0.0
            epoch_reg = 0.0

            pbar = tqdm(loader, desc=f'Epoch {epoch+1}')
            for batch in pbar:
                template = batch['template'].to(self.device)
                search = batch['search'].to(self.device)
                cls_target = batch['cls_target'].to(self.device)
                reg_target = batch['reg_target'].to(self.device)
                reg_mask = batch['reg_mask'].to(self.device)

                cls_pred, reg_pred = self.model(template, search)

                B, C, H, W = cls_pred.shape
                cls_target_flat = cls_target.permute(0, 2, 1, 3, 4)
                cls_target_flat = cls_target_flat.reshape(B, -1, H, W)

                B, C, H, W = reg_pred.shape
                reg_pred_flat = reg_pred.view(B, 5, 4, H, W).permute(0, 2, 1, 3, 4).reshape(B, 4, -1)

                total, cls_loss, reg_loss = self.criterion(
                    cls_pred, reg_pred_flat, cls_target_flat, reg_target, reg_mask
                )

                self.optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optimizer.step()

                epoch_loss += total.item()
                epoch_cls += cls_loss.item()
                epoch_reg += reg_loss.item()
                pbar.set_postfix(loss=total.item(), cls=cls_loss.item(), reg=reg_loss.item())

            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            avg_loss = epoch_loss / len(loader)
            avg_cls = epoch_cls / len(loader)
            avg_reg = epoch_reg / len(loader)
            logger.info(f'Epoch {epoch+1}: loss={avg_loss:.4f} cls={avg_cls:.4f} reg={avg_reg:.4f}')

            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': avg_loss,
                }, os.path.join(checkpoint_dir, 'best_model.pth'))
                logger.info(f'Checkpoint saved (loss={avg_loss:.4f})')

            if (epoch + 1) % self.cfg['train'].get('save_every', 10) == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': avg_loss,
                }, os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch+1}.pth'))

        torch.save({
            'epoch': self.cfg['train']['epochs'] - 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': avg_loss,
        }, os.path.join(checkpoint_dir, 'final_model.pth'))
        logger.info('Training complete.')

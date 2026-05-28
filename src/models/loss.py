import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_l1_loss(pred, target, beta=1.0):
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss.mean()


class SiamRPNLoss(nn.Module):
    def __init__(self, cls_weight=1.0, reg_weight=1.2):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight

    def forward(self, cls_pred, reg_pred, cls_target, reg_target, reg_mask):
        cls_loss = F.binary_cross_entropy_with_logits(cls_pred, cls_target, reduction='mean')

        reg_loss = smooth_l1_loss(reg_pred, reg_target)
        reg_loss = (reg_loss * reg_mask).mean()

        total = self.cls_weight * cls_loss + self.reg_weight * reg_loss
        return total, cls_loss, reg_loss

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone_mobilenet import MobileNetV2Backbone
from .fpn import FPN
from .head import DepthwiseCorr, RPNHead


class LightSiamRPN(nn.Module):
    def __init__(self, backbone_cfg=None, fpn_cfg=None, head_cfg=None):
        super().__init__()
        backbone_cfg = backbone_cfg or {}
        fpn_cfg = fpn_cfg or {}
        head_cfg = head_cfg or {}

        self.backbone = MobileNetV2Backbone(
            pretrained=backbone_cfg.get('pretrained', True),
            frozen=backbone_cfg.get('frozen', True)
        )

        in_channels = self.backbone.out_channels

        self.fpn = FPN(
            in_channels=in_channels,
            out_channels=fpn_cfg.get('out_channels', 256)
        )
        fpn_ch = self.fpn.out_channels

        self.corr = DepthwiseCorr(in_channels=fpn_ch)

        if fpn_cfg.get('multi_level_train', True):
            self.corr_weights = nn.Parameter(torch.ones(3) / 3)

        self.rpn_head = RPNHead(
            in_channels=fpn_ch,
            num_anchors=head_cfg.get('num_anchors', 5)
        )

    def forward(self, template, search):
        t_c3, t_c4, t_c5 = self.backbone(template)
        s_c3, s_c4, s_c5 = self.backbone(search)

        t_p3, t_p4, t_p5 = self.fpn(t_c3, t_c4, t_c5)
        s_p3, s_p4, s_p5 = self.fpn(s_c3, s_c4, s_c5)

        if self.training:
            corr3 = self.corr(t_p3, s_p3)
            corr4 = self.corr(t_p4, s_p4)
            corr5 = self.corr(t_p5, s_p5)

            target_size = corr4.shape[-2:]
            corr3 = F.interpolate(corr3, size=target_size, mode='bilinear', align_corners=False)
            corr5 = F.interpolate(corr5, size=target_size, mode='bilinear', align_corners=False)

            w = F.softmax(self.corr_weights, dim=0)
            corr = w[0] * corr3 + w[1] * corr4 + w[2] * corr5
        else:
            corr = self.corr(t_p4, s_p4)

        return self.rpn_head(corr)

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

BACKBONE_STRIDES = [8, 16, 32]
BACKBONE_CHANNELS = [32, 96, 320]


class MobileNetV2Backbone(nn.Module):
    def __init__(self, pretrained=True, frozen=True):
        super().__init__()
        if pretrained:
            self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        else:
            self.backbone = mobilenet_v2(weights=None)

        self.out_channels = BACKBONE_CHANNELS
        self.out_strides = BACKBONE_STRIDES

        self._freeze_backbone(frozen)

    def _freeze_backbone(self, frozen):
        for param in self.backbone.parameters():
            param.requires_grad = not frozen

    def forward(self, x):
        outs = []
        for i, module in enumerate(self.backbone.features):
            x = module(x)
            if i in {6, 13, 17}:
                outs.append(x)
        c3, c4, c5 = outs
        return c3, c4, c5

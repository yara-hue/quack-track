import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBN(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.conv(x))


class DepthwiseCorr(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.pre_conv = ConvBN(in_channels, in_channels, kernel_size=3)
        self.in_channels = in_channels

    def forward(self, template, search):
        template = self.pre_conv(template)
        search = self.pre_conv(search)

        B, C, Ht, Wt = template.shape
        _, _, Hs, Ws = search.shape

        corr = []
        for b in range(B):
            t = template[b:b+1]
            s = search[b:b+1]
            weight = t.view(C, 1, Ht, Wt)
            out = F.conv2d(s, weight, groups=C)
            corr.append(out)
        return torch.cat(corr, dim=0)


class RPNHead(nn.Module):
    def __init__(self, in_channels=256, num_anchors=5):
        super().__init__()
        self.cls_head = nn.Conv2d(in_channels, 2 * num_anchors, kernel_size=1)
        self.reg_head = nn.Conv2d(in_channels, 4 * num_anchors, kernel_size=1)
        self.num_anchors = num_anchors

    def forward(self, x):
        cls_logits = self.cls_head(x)
        reg_deltas = self.reg_head(x)
        return cls_logits, reg_deltas

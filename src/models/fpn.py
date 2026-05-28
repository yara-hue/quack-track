import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None):
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )


class FPN(nn.Module):
    def __init__(self, in_channels=(32, 96, 320), out_channels=256):
        super().__init__()
        c3_in, c4_in, c5_in = in_channels

        self.lateral_c3 = nn.Conv2d(c3_in, out_channels, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(c4_in, out_channels, kernel_size=1)
        self.lateral_c5 = nn.Conv2d(c5_in, out_channels, kernel_size=1)

        self.smooth_p4 = ConvBNReLU(out_channels, out_channels, kernel_size=3)
        self.smooth_p3 = ConvBNReLU(out_channels, out_channels, kernel_size=3)

        self.out_channels = out_channels

    def forward(self, c3, c4, c5):
        p5 = self.lateral_c5(c5)

        p4_up = F.interpolate(p5, size=c4.shape[-2:], mode='bilinear', align_corners=False)
        p4 = self.lateral_c4(c4) + p4_up
        p4 = self.smooth_p4(p4)

        p3_up = F.interpolate(p4, size=c3.shape[-2:], mode='bilinear', align_corners=False)
        p3 = self.lateral_c3(c3) + p3_up
        p3 = self.smooth_p3(p3)

        return p3, p4, p5

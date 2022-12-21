import math

from torch.autograd import Variable
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Attention import SpatialAttention, ChannelAttention, se_block


class ResDoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(ResDoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.Conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.BN1 = nn.BatchNorm2d(mid_channels)
        self.ReLU1 = nn.ReLU(inplace=True)
        self.Conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.BN2 = nn.BatchNorm2d(out_channels)
        self.ReLU2 = nn.ReLU(inplace=True)
        self.Conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.BN3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.ReLU1(self.BN1(self.Conv1(x)))
        x2 = self.BN2(self.Conv2(x1))
        out = self.ReLU2(self.BN3(self.Conv3(x)) + x2)
        return out


class ASPP(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        kernel_sizes = [1, 3, 3, 1]
        dilations = [1, 3, 6, 1]
        paddings = [0, 3, 6, 0]
        self.aspp = torch.nn.ModuleList()
        for aspp_idx in range(len(kernel_sizes)):
            conv = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_sizes[aspp_idx],
                stride=1,
                dilation=dilations[aspp_idx],
                padding=paddings[aspp_idx],
                bias=True)
            self.aspp.append(conv)
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.aspp_num = len(kernel_sizes)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0)

    def forward(self, x):
        avg_x = self.gap(x)
        out = []
        for aspp_idx in range(self.aspp_num):
            inp = avg_x if (aspp_idx == self.aspp_num - 1) else x
            out.append(F.relu_(self.aspp[aspp_idx](inp)))
        out[-1] = out[-1].expand_as(out[-2])
        out = torch.cat(out, dim=1)

        return out


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


def downsample():
    return nn.MaxPool2d(kernel_size=2, stride=2)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class FPN_conv(nn.Module):  # （32,64,128,256,512,）
    def __init__(self, in_channel):
        super(FPN_conv, self).__init__()

        # Bottom-up layers
        self.downsample = downsample()
        self.conv1 = conv_block(in_channel, 32)
        self.conv2 = conv_block(32, 64)
        self.conv3 = conv_block(64, 128)
        self.conv4 = conv_block(128, 256)
        self.conv5 = conv_block(256, 512)

        # Smooth layers
        self.smooth1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer2 = nn.Conv2d(64, 32, kernel_size=1)
        self.latlayer3 = nn.Conv2d(128, 64, kernel_size=1)
        self.latlayer4 = nn.Conv2d(256, 128, kernel_size=1)
        self.latlayer5 = nn.Conv2d(512, 256, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        # Bottom-up
        c1 = self.conv1(x)
        down1 = self.downsample(c1)
        c2 = self.conv2(down1)
        down2 = self.downsample(c2)
        c3 = self.conv3(down2)
        down3 = self.downsample(c3)
        c4 = self.conv4(down3)

        down4 = self.downsample(c4)
        c5 = self.conv5(down4)

        # Top-down
        p4 = self._upsample_add(self.latlayer5(c5), c4)
        p3 = self._upsample_add(self.latlayer4(p4), c3)
        p2 = self._upsample_add(self.latlayer3(p3), c2)
        p1 = self._upsample_add(self.latlayer2(p2), c1)
        # Smooth
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)
        p1 = self.smooth1(p1)

        return p1, p2, p3, p4, c5


class FPN_ASPP(nn.Module):  # （32,64,128,256,512,）
    def __init__(self, in_channel):
        super(FPN_ASPP, self).__init__()

        # Bottom-up layers
        self.downsample = downsample()
        self.conv1 = conv_block(in_channel, 32)
        self.conv2 = conv_block(32, 64)
        self.conv3 = conv_block(64, 128)
        self.conv4 = conv_block(128, 256)
        self.conv5 = conv_block(256, 512)

        # ASPP layers
        self.aspp1 = ASPP(32, 8)
        self.aspp2 = ASPP(64, 16)
        self.aspp3 = ASPP(128, 32)
        self.aspp4 = ASPP(256, 64)
        self.aspp5 = ASPP(512, 128)

        # Smooth layers
        self.smooth1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer2 = nn.Conv2d(64, 32, kernel_size=1)
        self.latlayer3 = nn.Conv2d(128, 64, kernel_size=1)
        self.latlayer4 = nn.Conv2d(256, 128, kernel_size=1)
        self.latlayer5 = nn.Conv2d(512, 256, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)

        # fusion layers
        self.cha_att1 = se_block(32)
        self.cha_att2 = se_block(64)
        self.cha_att3 = se_block(128)
        self.cha_att4 = se_block(256)
        self.cha_att5 = se_block(512)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        # Bottom-up
        c1 = self.conv1(x)
        down1 = self.downsample(c1)
        c2 = self.conv2(down1)
        down2 = self.downsample(c2)
        c3 = self.conv3(down2)
        down3 = self.downsample(c3)
        c4 = self.conv4(down3)

        down4 = self.downsample(c4)
        c5 = self.conv5(down4)

        # Top-down1
        p4 = self._upsample_add(self.latlayer5(c5), c4)
        p3 = self._upsample_add(self.latlayer4(p4), c3)
        p2 = self._upsample_add(self.latlayer3(p3), c2)
        p1 = self._upsample_add(self.latlayer2(p2), c1)

        # ASPP
        a5 = self.aspp5(c5)
        a4 = self.aspp4(p4)
        a3 = self.aspp3(p3)
        a2 = self.aspp2(p2)
        a1 = self.aspp1(p1)

        # Smooth
        a5 = self.smooth5(a5)
        a4 = self.smooth4(a4)
        a3 = self.smooth3(a3)
        a2 = self.smooth2(a2)
        a1 = self.smooth1(a1)

        # add
        a5 = a5 + c5
        a4 = a4 + c4
        a3 = a3 + c3
        a2 = a2 + c2
        a1 = a1 + c1

        # Top-down2
        a4 = self._upsample_add(self.latlayer5(a5), a4)
        a3 = self._upsample_add(self.latlayer4(a4), a3)
        a2 = self._upsample_add(self.latlayer3(a3), a2)
        a1 = self._upsample_add(self.latlayer2(a2), a1)

        # fusion layers
        w1 = self.cha_att1(a1)
        w2 = self.cha_att2(a2)
        w3 = self.cha_att3(a3)
        w4 = self.cha_att4(a4)
        w5 = self.cha_att5(a5)

        a1 = a1 * w1 + c1 * (1 - w1)
        a2 = a2 * w2 + c2 * (1 - w2)
        a3 = a3 * w3 + c3 * (1 - w3)
        a4 = a4 * w4 + c4 * (1 - w4)
        a5 = a5 * w5 + c5 * (1 - w5)
        return a1, a2, a3, a4, a5, c5


class FPN_Res_ASPP(nn.Module):  # （32,64,128,256,512,）
    def __init__(self, in_channel):
        super(FPN_Res_ASPP, self).__init__()

        # Bottom-up layers
        self.downsample = downsample()
        self.conv1 = ResDoubleConv(in_channel, 32)
        self.conv2 = ResDoubleConv(32, 64)
        self.conv3 = ResDoubleConv(64, 128)
        self.conv4 = ResDoubleConv(128, 256)
        self.conv5 = ResDoubleConv(256, 512)

        # ASPP layers
        self.aspp1 = ASPP(32, 8)
        self.aspp2 = ASPP(64, 16)
        self.aspp3 = ASPP(128, 32)
        self.aspp4 = ASPP(256, 64)
        self.aspp5 = ASPP(512, 128)

        # Smooth layers
        self.smooth1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer2 = nn.Conv2d(64, 32, kernel_size=1)
        self.latlayer3 = nn.Conv2d(128, 64, kernel_size=1)
        self.latlayer4 = nn.Conv2d(256, 128, kernel_size=1)
        self.latlayer5 = nn.Conv2d(512, 256, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)

        # fusion layers
        self.cha_att1 = se_block(32)
        self.cha_att2 = se_block(64)
        self.cha_att3 = se_block(128)
        self.cha_att4 = se_block(256)
        self.cha_att5 = se_block(512)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        # Bottom-up
        c1 = self.conv1(x)
        down1 = self.downsample(c1)
        c2 = self.conv2(down1)
        down2 = self.downsample(c2)
        c3 = self.conv3(down2)
        down3 = self.downsample(c3)
        c4 = self.conv4(down3)

        down4 = self.downsample(c4)
        c5 = self.conv5(down4)

        # Top-down1
        p4 = self._upsample_add(self.latlayer5(c5), c4)
        p3 = self._upsample_add(self.latlayer4(p4), c3)
        p2 = self._upsample_add(self.latlayer3(p3), c2)
        p1 = self._upsample_add(self.latlayer2(p2), c1)

        # ASPP
        a5 = self.aspp5(c5)
        a4 = self.aspp4(p4)
        a3 = self.aspp3(p3)
        a2 = self.aspp2(p2)
        a1 = self.aspp1(p1)

        # Smooth
        a5 = self.smooth5(a5)
        a4 = self.smooth4(a4)
        a3 = self.smooth3(a3)
        a2 = self.smooth2(a2)
        a1 = self.smooth1(a1)

        # add
        a5 = a5 + c5
        a4 = a4 + c4
        a3 = a3 + c3
        a2 = a2 + c2
        a1 = a1 + c1

        # Top-down2
        a4 = self._upsample_add(self.latlayer5(a5), a4)
        a3 = self._upsample_add(self.latlayer4(a4), a3)
        a2 = self._upsample_add(self.latlayer3(a3), a2)
        a1 = self._upsample_add(self.latlayer2(a2), a1)

        # fusion layers
        w1 = self.cha_att1(a1)
        w2 = self.cha_att2(a2)
        w3 = self.cha_att3(a3)
        w4 = self.cha_att4(a4)
        w5 = self.cha_att5(a5)

        a1 = a1 * w1 + c1 * (1 - w1)
        a2 = a2 * w2 + c2 * (1 - w2)
        a3 = a3 * w3 + c3 * (1 - w3)
        a4 = a4 * w4 + c4 * (1 - w4)
        a5 = a5 * w5 + c5 * (1 - w5)
        return a1, a2, a3, a4, a5, c5

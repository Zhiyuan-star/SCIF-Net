""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


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


class DesDoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DesDoubleConv, self).__init__()
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
        x1 = self.BN1(self.Conv1(x))
        x2 = self.ReLU1(self.BN3(self.Conv3(x)) + x1)
        x3 = self.BN2(self.Conv2(x2))
        out = self.ReLU2(self.BN3(self.Conv3(x)) + x2 + x3)
        return out


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class ResDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class DesDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DesDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

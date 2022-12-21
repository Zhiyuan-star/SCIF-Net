from torch import nn
import torch
from model.FPN import FPN_ASPP, FPN_Res_ASPP
from model.mult_attention import mult_att
from model.Attention import ChannelAttention, SpatialAttention
from model.unet_parts import *
import torch.nn.functional as F
from functools import partial

nonlinearity = partial(F.relu, inplace=True)


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


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        return x * psi


class CIF_block(nn.Module):
    def __init__(self, in_channels):
        super(CIF_block, self).__init__()
        self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.spa = SpatialAttention(kernel_size=3)
        self.cha = ChannelAttention(in_planes=in_channels)

    def forward(self, l, h):
        cat = torch.cat((l, h), dim=1)
        cat = self.conv(cat)
        spa_cat = self.spa(cat)
        l = l * spa_cat
        h = h * self.cha(l)
        out = l + h
        return out


class ourNet1(nn.Module):  # Baseline
    def __init__(self, in_channels, n_classes, bilinear=True):
        super(ourNet1, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits = torch.sigmoid(logits)
        return logits


class ourNet2(nn.Module):  # Baseline + FPE + MFDA
    def __init__(self, in_channels=3, n_classes=1):
        super(ourNet2, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.FPN = FPN_ASPP(in_channels)
        self.multAtt = mult_att(512, 512)
        self.Up5 = up_conv(ch_in=512, ch_out=256)

        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256, ch_out=128)

        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)

        self.Up3 = up_conv(ch_in=128, ch_out=64)

        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)

        self.Up2 = up_conv(ch_in=64, ch_out=32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32, n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1, x2, x3, x4, x5, x6 = self.FPN(x)
        att = self.multAtt(x6)

        # decoding + concat path
        d5 = self.Up5(att)

        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)

        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)

        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)

        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = torch.sigmoid(d1)
        return d1


class ourNet3(nn.Module):  # Baseline + MFDA + ASIF
    def __init__(self, in_channels=3, n_classes=1):
        super(ourNet3, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        self.multAtt = mult_att(512, 512)
        self.Up5 = up_conv(ch_in=512, ch_out=256)
        self.Att5 = CIF_block(256)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256, ch_out=128)
        self.Att4 = CIF_block(128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)

        self.Up3 = up_conv(ch_in=128, ch_out=64)
        self.Att3 = CIF_block(64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)

        self.Up2 = up_conv(ch_in=64, ch_out=32)
        self.Att2 = CIF_block(32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32, n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        att = self.multAtt(x5)

        # decoding + concat path
        d5 = self.Up5(att)
        x4 = self.Att5(x4, d5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(x3, d4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(x2, d3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(x1, d2)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = torch.sigmoid(d1)
        return d1


class ourNet4(nn.Module):  # Baseline + FPE + ASIF
    def __init__(self, in_channels=3, n_classes=1):
        super(ourNet4, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.FPN = FPN_ASPP(in_channels)

        # self.multAtt = mult_att(512, 512)

        self.Up5 = up_conv(ch_in=512, ch_out=256)
        self.Att5 = CIF_block(256)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256, ch_out=128)
        self.Att4 = CIF_block(128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)

        self.Up3 = up_conv(ch_in=128, ch_out=64)
        self.Att3 = CIF_block(64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)

        self.Up2 = up_conv(ch_in=64, ch_out=32)
        self.Att2 = CIF_block(32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32, n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1, x2, x3, x4, x5, x6 = self.FPN(x)

        # decoding + concat path
        d5 = self.Up5(x6)
        x4 = self.Att5(x4, d5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(x3, d4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(x2, d3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(x1, d2)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = torch.sigmoid(d1)
        return d1


class ourNet5(nn.Module):  # Baseline + FPE + MFDA + ASIF
    def __init__(self, in_channels=3, n_classes=1):
        super(ourNet5, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.FPN = FPN_ASPP(in_channels)
        self.multAtt = mult_att(512, 512)
        self.Up5 = up_conv(ch_in=512, ch_out=256)
        self.Att5 = CIF_block(256)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256, ch_out=128)
        self.Att4 = CIF_block(128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)

        self.Up3 = up_conv(ch_in=128, ch_out=64)
        self.Att3 = CIF_block(64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)

        self.Up2 = up_conv(ch_in=64, ch_out=32)
        self.Att2 = CIF_block(32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32, n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1, x2, x3, x4, x5, x6 = self.FPN(x)
        att = self.multAtt(x6)

        # decoding + concat path
        d5 = self.Up5(att)
        x4 = self.Att5(x4, d5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(x3, d4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(x2, d3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(x1, d2)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = torch.sigmoid(d1)
        return d1


class ourNet6(nn.Module):  # Baseline + FPE
    def __init__(self, in_channels=3, n_classes=1):
        super(ourNet6, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.FPN = FPN_ASPP(in_channels)
        self.Up5 = up_conv(ch_in=512, ch_out=256)

        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256, ch_out=128)

        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)

        self.Up3 = up_conv(ch_in=128, ch_out=64)

        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)

        self.Up2 = up_conv(ch_in=64, ch_out=32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32, n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1, x2, x3, x4, x5, x6 = self.FPN(x)

        # decoding + concat path
        d5 = self.Up5(x6)

        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)

        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)

        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)

        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = torch.sigmoid(d1)
        return d1


class ourNet7(nn.Module):  # Baseline + ASIF
    def __init__(self, in_channels=3, n_classes=1):
        super(ourNet7, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        self.Up5 = up_conv(ch_in=512, ch_out=256)
        self.Att5 = CIF_block(256)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256, ch_out=128)
        self.Att4 = CIF_block(128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)

        self.Up3 = up_conv(ch_in=128, ch_out=64)
        self.Att3 = CIF_block(64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)

        self.Up2 = up_conv(ch_in=64, ch_out=32)
        self.Att2 = CIF_block(32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32, n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(x4, d5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(x3, d4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(x2, d3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(x1, d2)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = torch.sigmoid(d1)
        return d1

class ourNet8(nn.Module):  # Baseline(残差连接)
    def __init__(self, in_channels, n_classes, bilinear=True):
        super(ourNet8, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = ResDoubleConv(in_channels, 32)
        self.down1 = ResDown(32, 64)
        self.down2 = ResDown(64, 128)
        self.down3 = ResDown(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = ResDown(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits = torch.sigmoid(logits)
        return logits

class ourNet9(nn.Module):  # Baseline(密集连接)
    def __init__(self, in_channels, n_classes, bilinear=True):
        super(ourNet9, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DesDoubleConv(in_channels, 32)
        self.down1 = DesDown(32, 64)
        self.down2 = DesDown(64, 128)
        self.down3 = DesDown(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = DesDown(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits = torch.sigmoid(logits)
        return logits

class ourNet10(nn.Module):  # Baseline(残差连接) + FPE + MFDA + ASIF
    def __init__(self, in_channels=3, n_classes=1):
        super(ourNet10, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.FPN = FPN_Res_ASPP(in_channels)
        self.multAtt = mult_att(512, 512)
        self.Up5 = up_conv(ch_in=512, ch_out=256)
        self.Att5 = CIF_block(256)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256, ch_out=128)
        self.Att4 = CIF_block(128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)

        self.Up3 = up_conv(ch_in=128, ch_out=64)
        self.Att3 = CIF_block(64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)

        self.Up2 = up_conv(ch_in=64, ch_out=32)
        self.Att2 = CIF_block(32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32, n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1, x2, x3, x4, x5, x6 = self.FPN(x)
        att = self.multAtt(x6)

        # decoding + concat path
        d5 = self.Up5(att)
        x4 = self.Att5(x4, d5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(x3, d4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(x2, d3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(x1, d2)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = torch.sigmoid(d1)
        return d1

if __name__ == '__main__':
    x = torch.randn(2, 1, 512, 512)
    print(x.shape)
    Net = ourNet9(1, 1)
    x = Net(x)
    print(x.shape)

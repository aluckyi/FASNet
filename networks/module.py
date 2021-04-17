import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSFA(nn.Module):
    def __init__(self, in_channels, mid_channels=64, out_channels=16):
        super(MSFA, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True))
        self.msff = MSFF(mid_channels, mid_channels//2, out_channels)
        self.se = SEBottleneck(out_channels, out_channels//2, out_channels)
        # self.se = SEBottleneck(out_channels, out_channels // 4, out_channels, reduction=4)  # FASNet2_2

    def forward(self, x):
        x = self.conv1(x)
        x = self.msff(x)
        x = self.se(x)
        return x


class MSFF(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernels=(3, 5, 7, 9)):
        super(MSFF, self).__init__()
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(True))

        kernel1, kernel2, kernel3, kernel4 = tuple(kernels)
        self.b1 = nn.Sequential(SeparableConv2d(in_channels, mid_channels, kernel1),
                                nn.BatchNorm2d(mid_channels),
                                nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(SeparableConv2d(in_channels, mid_channels, kernel2),
                                nn.BatchNorm2d(mid_channels),
                                nn.ReLU(inplace=True))
        self.b3 = nn.Sequential(SeparableConv2d(in_channels, mid_channels, kernel3),
                                nn.BatchNorm2d(mid_channels),
                                nn.ReLU(inplace=True))
        self.b4 = nn.Sequential(SeparableConv2d(in_channels, mid_channels, kernel4),
                                nn.BatchNorm2d(mid_channels),
                                nn.ReLU(inplace=True))
        self.project = nn.Sequential(
            nn.Conv2d(5 * mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        x = torch.cat((feat0, feat1, feat2, feat3, feat4), dim=1)
        x = self.project(x)
        return x


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super(SeparableConv2d, self).__init__()
        stride = 1
        pad = (kernel_size - 1) // 2
        dilation = 1

        self.conv1 = nn.Conv2d(in_channels, in_channels, (kernel_size, 1), stride, (pad, 0), dilation,
                               groups=in_channels, bias=bias)
        self.conv2 = nn.Conv2d(in_channels, in_channels, (1, kernel_size), stride, (0, pad), dilation,
                               groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pointwise(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class CSELayer(nn.Module):
    def __init__(self, in_channels, reduction=2):
        super(CSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        x = self.avg_pool(inputs)
        x = self.fc(x)
        return inputs * x.expand_as(inputs)


class SEBottleneck(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, stride=1, downsample=None, reduction=2):
        super(SEBottleneck, self).__init__()
        self.conv1 = conv1x1(in_channels, mid_channels)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = conv3x3(mid_channels, mid_channels, stride)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = conv1x1(mid_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.se = CSELayer(out_channels, reduction)

        self.downsample = downsample
        self.stride = stride

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, relu6=False, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs))
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

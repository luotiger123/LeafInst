"""
Dynamic Head Module for Feature Enhancement

This file implements a dynamic feature enhancement module designed for
dense prediction tasks such as object detection or instance segmentation.
This implementation is built upon the OpenMMLab framework.

License
-------
This project follows the license of OpenMMLab components where applicable.
"""

# Copyright (c) OpenMMLab. All rights reserved.

from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor
from mmcv.cnn import ConvModule
from torchvision.ops import DeformConv2d


class BasicConv(nn.Module):
    """
    Basic convolution block with optional BN and ReLU.

    Parameters
    ----------
    in_planes : int
        Number of input channels
    out_planes : int
        Number of output channels
    kernel_size : int or tuple
        Convolution kernel size
    """

    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super().__init__()

        if padding is None:
            if isinstance(kernel_size, tuple):
                padding = (
                    (kernel_size[0] - 1) // 2 if stride == 1 else 0,
                    (kernel_size[1] - 1) // 2 if stride == 1 else 0,
                )
            else:
                padding = (kernel_size - 1) // 2 if stride == 1 else 0

        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)

        if self.bn is not None:
            x = self.bn(x)

        if self.relu is not None:
            x = self.relu(x)

        return x


class DepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution.

    This block decomposes standard convolution into:
    - depthwise convolution
    - pointwise convolution
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=None,
        bias=True,
        norm_layer=None,
        activation=None,
    ):
        super().__init__()

        groups = in_channels if groups is None else groups
        assert groups == in_channels

        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=groups,
            bias=bias,
        )

        self.pointwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )

        self.norm = norm_layer(out_channels) if norm_layer else None
        self.activation = activation() if isinstance(activation, type) else activation

    def forward(self, x: Tensor) -> Tensor:

        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm is not None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class DynamicHead(nn.Module):
    """
    Dynamic feature enhancement head.

    This module integrates multiple feature transformation branches:
    - Direction-aware convolution
    - Standard convolution
    - Deformable convolution

    Final features are fused using residual aggregation.
    """

    def __init__(self, in_channels: int):
        super().__init__()

        self.in_channel = in_channels
        k = 3
        inter_planes = in_channels // 2

        # Feature expansion
        self.upChannels = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels, in_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels * 4),
            nn.ReLU(inplace=True),
        )

        # Direction-aware convolutions
        self.daspH2V = nn.Sequential(
            BasicConv(in_channels, inter_planes, kernel_size=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, k)),
            BasicConv((inter_planes // 2) * 3, in_channels, kernel_size=(k, 1)),
            BasicConv(in_channels, in_channels, kernel_size=3),
        )

        self.daspV2H = nn.Sequential(
            BasicConv(in_channels, inter_planes, kernel_size=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(k, 1)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, k)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3),
        )

        # Standard convolution branch
        self.common = nn.Sequential(
            BasicConv(in_channels, inter_planes, kernel_size=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=3),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3),
        )

        # Deformable convolution branch
        self.dcn = nn.Sequential(
            ConvModule(
                in_channels,
                inter_planes,
                3,
                stride=1,
                padding=1,
                conv_cfg=dict(type="DCNv2"),
                norm_cfg=dict(type="BN"),
                bias=True,
            ),
            BasicConv(inter_planes, in_channels, kernel_size=1),
        )

        # Channel reduction
        self.downchannels = nn.Sequential(
            nn.Conv2d(4 * in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:

        x_upchannels = self.upChannels(x)

        x_up1, x_up2, x_up3, x_up4 = torch.split(
            x_upchannels, self.in_channel, dim=1
        )

        x_up1 = self.daspH2V(x_up1)
        x_up2 = self.daspV2H(x_up2)
        x_up3 = self.common(x_up3)
        x_up4 = self.dcn(x_up4)

        x_f = torch.cat([x_up1, x_up2, x_up3, x_up4], dim=1)

        x_f = 0.3 * x_upchannels + x_f
        x_f = self.downchannels(x_f)

        x_merge = 0.3 * x + x_f

        return x_merge
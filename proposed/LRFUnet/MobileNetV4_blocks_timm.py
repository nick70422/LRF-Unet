from typing import Callable, Dict, Optional, Type

import torch
import torch.nn as nn
from torch.nn import functional as F

from timm.layers import DropPath, make_divisible, LayerType, ConvNormAct

ModuleType = Type[nn.Module]

def num_groups(group_size: Optional[int], channels: int):
    if not group_size:  # 0 or None
        return 1  # normal conv with 1 group
    else:
        # NOTE group_size == 1 -> depthwise conv
        assert channels % group_size == 0
        return channels // group_size

class LayerScale2d(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma.view(1, -1, 1, 1)
        return x.mul_(gamma) if self.inplace else x * gamma

class UniversalInvertedResidual(nn.Module):
    """ Universal Inverted Residual Block (aka Universal Inverted Bottleneck, UIB)

    For MobileNetV4 - https://arxiv.org/abs/, referenced from
    https://github.com/tensorflow/models/blob/d93c7e932de27522b2fa3b115f58d06d6f640537/official/vision/modeling/layers/nn_blocks.py#L778
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            dw_kernel_size_start: int = 0,
            dw_kernel_size_mid: int = 3,
            dw_kernel_size_end: int = 0,
            stride: int = 1,
            dilation: int = 1,
            group_size: int = 1,
            pad_type: str = '',
            noskip: bool = False,
            exp_ratio: float = 1.0,
            act_layer: LayerType = nn.ReLU,
            norm_layer: LayerType = nn.BatchNorm2d,
            aa_layer: Optional[LayerType] = None,
            se_layer: Optional[ModuleType] = None,
            conv_kwargs: Optional[Dict] = None,
            drop_path_rate: float = 0.,
            layer_scale_init_value: Optional[float] = 1e-5,
    ):
        super(UniversalInvertedResidual, self).__init__()
        conv_kwargs = conv_kwargs or {}
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip
        if stride > 1:
            assert dw_kernel_size_start or dw_kernel_size_mid or dw_kernel_size_end

        # FIXME dilation isn't right w/ extra ks > 1 convs
        if dw_kernel_size_start:
            dw_start_stride = stride if not dw_kernel_size_mid else 1
            dw_start_groups = num_groups(group_size, in_chs)
            self.dw_start = ConvNormAct(
                in_chs, in_chs, dw_kernel_size_start,
                stride=dw_start_stride,
                dilation=dilation,  # FIXME
                groups=dw_start_groups,
                padding=pad_type,
                apply_act=False,
                act_layer=act_layer,
                norm_layer=norm_layer,
                aa_layer=aa_layer,
                **conv_kwargs,
            )
        else:
            self.dw_start = nn.Identity()

        # Point-wise expansion
        mid_chs = make_divisible(in_chs * exp_ratio)
        self.pw_exp = ConvNormAct(
            in_chs, mid_chs, 1,
            padding=pad_type,
            act_layer=act_layer,
            norm_layer=norm_layer,
            **conv_kwargs,
        )

        # Middle depth-wise convolution
        if dw_kernel_size_mid:
            groups = num_groups(group_size, mid_chs)
            self.dw_mid = ConvNormAct(
                mid_chs, mid_chs, dw_kernel_size_mid,
                stride=stride,
                dilation=dilation,  # FIXME
                groups=groups,
                padding=pad_type,
                act_layer=act_layer,
                norm_layer=norm_layer,
                aa_layer=aa_layer,
                **conv_kwargs,
            )
        else:
            # keeping mid as identity so it can be hooked more easily for features
            self.dw_mid = nn.Identity()

        # Squeeze-and-excitation
        self.se = se_layer(mid_chs, act_layer=act_layer) if se_layer else nn.Identity()

        # Point-wise linear projection
        self.pw_proj = ConvNormAct(
            mid_chs, out_chs, 1,
            padding=pad_type,
            apply_act=False,
            act_layer=act_layer,
            norm_layer=norm_layer,
            **conv_kwargs,
        )

        if dw_kernel_size_end:
            dw_end_stride = stride if not dw_kernel_size_start and not dw_kernel_size_mid else 1
            dw_end_groups = num_groups(group_size, out_chs)
            if dw_end_stride > 1:
                assert not aa_layer
            self.dw_end = ConvNormAct(
                out_chs, out_chs, dw_kernel_size_end,
                stride=dw_end_stride,
                dilation=dilation,
                groups=dw_end_groups,
                padding=pad_type,
                apply_act=False,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **conv_kwargs,
            )
        else:
            self.dw_end = nn.Identity()

        if layer_scale_init_value is not None:
            self.layer_scale = LayerScale2d(out_chs, layer_scale_init_value)
        else:
            self.layer_scale = nn.Identity()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PWL
            return dict(module='pw_proj.conv', hook_type='forward_pre', num_chs=self.pw_proj.conv.in_channels)
        else:  # location == 'bottleneck', block output
            return dict(module='', num_chs=self.pw_proj.conv.out_channels)

    def forward(self, x):
        shortcut = x
        x = self.dw_start(x)
        x = self.pw_exp(x)
        x = self.dw_mid(x)
        x = self.se(x)
        x = self.pw_proj(x)
        x = self.dw_end(x)
        x = self.layer_scale(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x
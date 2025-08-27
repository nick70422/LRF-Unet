import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, TypeVar, Union, Tuple
import collections
from itertools import repeat

def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse

_single = _ntuple(1, "_single")
_pair = _ntuple(2, "_pair")
_triple = _ntuple(3, "_triple")
_quadruple = _ntuple(4, "_quadruple")

T = TypeVar('T')
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
_size_2_t = _scalar_or_tuple_2_t[int]

class ConvLRF(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: Union[str, _size_2_t] = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None, r=16):
        super(ConvLRF, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        #print(self.__dict__.keys())
        
        self.r = r
        
        # Placeholder for A and B matrices
        #print(f'self.r: {self.r}')
        #print(f'self.kernel_size[0]: {self.kernel_size[0]}')
        #print(f'self.in_channels: {self.in_channels}')
        #print(f'self.out_channels: {self.out_channels}')
        
        self.A = nn.Parameter(torch.randn(self.r * self.kernel_size[0], self.in_channels * self.kernel_size[0]))
        self.B = nn.Parameter(torch.randn(self.out_channels * self.kernel_size[0], self.r * self.kernel_size[0]))
        #self.A = nn.Parameter(torch.randn(self.r, self.in_channels * self.kernel_size[0]))
        #self.B = nn.Parameter(torch.randn(self.out_channels * self.kernel_size[0], self.r))

        self.reset_LRFparameters()

    def reset_LRFparameters(self):
        # Initialize A and B matrices
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        #nn.init.zeros_(self.B)

    def forward(self, x):        
        # Generate the weight matrix using A and B
        weight = (self.B @ self.A).view(self.out_channels, self.in_channels, *self.kernel_size)
        
        # Perform the convolution
        return self._conv_forward(x, weight, self.bias)
    
class ConvLRF2(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: Union[str, _size_2_t] = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None, r=16):
        super(ConvLRF2, self).__init__()
        #print(self.__dict__.keys())
        self.conv1 = nn.Conv2d(in_channels, r, (1, 1), 1, (0, 0), dilation, groups, bias, padding_mode, device, dtype)
        #self.bn1 = nn.BatchNorm2d(r)
        #self.activation1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(r, out_channels, kernel_size, stride, ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2), dilation, groups, bias, padding_mode, device, dtype)
        #self.conv2 = nn.Conv2d(r, out_channels, kernel_size, stride, ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2), dilation, r, bias, padding_mode, device, dtype)
        #self.conv2 = nn.Conv2d(r, out_channels, kernel_size, stride, ((kernel_size[0] - 1) * 9 // 2, (kernel_size[1] - 1) * 9 // 2), 9, groups, bias, padding_mode, device, dtype)

        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x):        
        # Perform the convolution
        x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.activation1(x)
        x = self.conv2(x)

        return x
    
class ConvLRF2_dilated(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: Union[str, _size_2_t] = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None, r=16):
        super(ConvLRF2_dilated, self).__init__()
        #print(self.__dict__.keys())
        self.conv1 = nn.Conv2d(in_channels, r, (1, 1), 1, (0, 0), dilation, groups, bias, padding_mode, device, dtype)
        
        # conv21 with dilation rate 1
        dilation1 = 1
        padding1 = ((kernel_size[0] - 1) // 2 * dilation1, (kernel_size[1] - 1) // 2 * dilation1)
        self.conv21 = nn.Conv2d(r, r, kernel_size, stride, padding1, dilation1, r, bias, padding_mode, device, dtype)
        
        # conv22 with dilation rate 3
        dilation2 = 3
        padding2 = ((kernel_size[0] - 1) // 2 * dilation2, (kernel_size[1] - 1) // 2 * dilation2)
        self.conv22 = nn.Conv2d(r, r, kernel_size, stride, padding2, dilation2, r, bias, padding_mode, device, dtype)
        
        # conv23 with dilation rate 5
        dilation3 = 5
        padding3 = ((kernel_size[0] - 1) // 2 * dilation3, (kernel_size[1] - 1) // 2 * dilation3)
        self.conv23 = nn.Conv2d(r, r, kernel_size, stride, padding3, dilation3, r, bias, padding_mode, device, dtype)

        self.conv3 = nn.Conv2d(3 * r, out_channels, (1, 1), 1, (0, 0), dilation, groups, bias, padding_mode, device, dtype)
        self.convRes = nn.Conv2d(in_channels, out_channels, (1, 1), stride, (0, 0), dilation, groups, bias, padding_mode, device, dtype)

        self.conv1.reset_parameters()
        self.conv21.reset_parameters()
        self.conv22.reset_parameters()
        self.conv23.reset_parameters()
        self.conv3.reset_parameters()
        self.convRes.reset_parameters()

        #print("#" * 10)
        #print("stride: ", stride)
        #print("padding1: ", padding1[0])
        #print("padding2: ", padding2[0])
        #print("padding3: ", padding3[0])

    def forward(self, x):
        conv1x = self.conv1(x)
        x1 = self.conv21(conv1x)
        x2 = self.conv22(conv1x)
        x3 = self.conv23(conv1x)
        
        x_concat = torch.cat((x1, x2, x3), dim=1)
        
        # 輸入到conv3
        x = self.conv3(x_concat) + self.convRes(x)

        return x
    
if __name__ == '__main__':
    from nnunet.network_architecture.initialization import InitWeights_He
    import numpy as np
    feat = torch.randn(2, 64, 512, 512).cuda()
    conv1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1, dilation=1, bias=True).cuda()
    conv2 = ConvLRF2(64, 128, kernel_size=(3, 3), stride=1, padding=1, dilation=1, bias=True,
                    r=64).cuda()
    conv3 = ConvLRF2_dilated(64, 128, kernel_size=(3, 3), stride=1, padding=1, dilation=1, bias=True,
                    r=64).cuda()
    #delattr(conv2, 'weight')
    
    InitWeights_He(1e-2)(conv1)

    model_parameters = filter(lambda p: p[1].requires_grad, (conv1).named_parameters())
    param_sum = 0
    for name, param in model_parameters:
        print(f"Parameter name: {name}, requires_grad: {param.requires_grad}")
        print(param_sum)
        param_sum += np.prod(param.size())
    print('No of params ', param_sum)

    model_parameters = filter(lambda p: p[1].requires_grad, (conv2).named_parameters())
    param_sum = 0
    for name, param in model_parameters:
        print(f"Parameter name: {name}, requires_grad: {param.requires_grad}")
        print(param_sum)
        param_sum += np.prod(param.size())
    print('No of params ', param_sum)
    
    model_parameters = filter(lambda p: p[1].requires_grad, (conv3).named_parameters())
    param_sum = 0
    for name, param in model_parameters:
        print(f"Parameter name: {name}, requires_grad: {param.requires_grad}")
        print(param_sum)
        param_sum += np.prod(param.size())
    print('No of params ', param_sum)
    
    #state_dict = conv1.state_dict()
    #conv2.load_state_dict(state_dict, strict=False)

    print(feat.shape)
    #out1 = conv1(feat)
    #out2 = conv2(feat)
    out3 = conv3(feat)
    ##print(out1.shape)
    print(out3.shape)
#
    #if torch.allclose(out1, out2, atol=1e-6):
    #    print("out1 and out2 have the same values.")
    #else:
    #    print("out1 and out2 have different values.")
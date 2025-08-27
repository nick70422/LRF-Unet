import torch
import torch.nn as nn
import math
from typing import TypeVar, Union, Tuple
import collections
from itertools import repeat
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tucker_tensor import tucker_to_tensor

# 将 Tensorly 设置为使用 PyTorch 作为后端
tl.set_backend('pytorch')

def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse

T = TypeVar('T')
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
_size_2_t = _scalar_or_tuple_2_t[int]

def choose_ranks(tensor, energy_threshold=0.95):
    ranks = []
    for mode in range(tensor.ndim):
        unfolded = tl.unfold(tensor, mode)
        _, S, _ = torch.svd(unfolded)
        cumulative_energy = torch.cumsum(S**2, dim=0) / torch.sum(S**2)
        rank = torch.searchsorted(cumulative_energy, energy_threshold).item() + 1
        ranks.append(rank)
    return ranks

class ConvTucker(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: Union[str, _size_2_t] = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None, r=16):
        super(ConvTucker, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)

    def tucker_factorization(self, energy_threshold):        
        ranks = choose_ranks(self.weight, energy_threshold=energy_threshold)
        self.core, factors = tucker(self.weight, rank=ranks)
        self.core = nn.Parameter(self.core)
        self.factor0 = nn.Parameter(factors[0])
        self.factor1 = nn.Parameter(factors[1])
        self.factor2 = nn.Parameter(factors[2])
        self.factor3 = nn.Parameter(factors[3])
        
        del self.weight

    def forward(self, x):
        factor0 = self.factor0
        factor1 = self.factor1
        factor2 = self.factor2
        factor3 = self.factor3
        weight = tucker_to_tensor((self.core, [factor0, factor1, factor2, factor3]))
        
        # Perform the convolution
        return self._conv_forward(x, weight, self.bias)

if __name__ == '__main__':
    import numpy as np
    #
    feat = torch.randn(2, 64, 512, 512).cuda()
    conv1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1, dilation=1, bias=True).cuda()
    conv2 = ConvTucker(64, 128, kernel_size=(3, 3), stride=1, padding=1, dilation=1, bias=True,
                    r=64).cuda()
    conv2.tucker_factorization(0.95)
    #
    model_parameters = filter(lambda p: p[1].requires_grad, (conv1).named_parameters())
    param_sum = 0
    for name, param in model_parameters:
        print(f"Parameter name: {name}, requires_grad: {param.requires_grad}")
        print(param_sum)
        param_sum += np.prod(param.size())
    print('No of params ', param_sum)
    #
    model_parameters = filter(lambda p: p[1].requires_grad, (conv2).named_parameters())
    param_sum = 0
    for name, param in model_parameters:
        print(f"Parameter name: {name}, requires_grad: {param.requires_grad}")
        print("shape:", param.shape)
        print("param_sum:", param_sum)
        param_sum += np.prod(param.size())
    print('No of params ', param_sum)
    #
    print(feat.shape)
    #out1 = conv1(feat)
    out2 = conv2(feat)
    ##print(out1.shape)
    print(out2.shape)
    ## 假设卷积核为四维张量
    #conv_weights = torch.rand(32, 128, 3, 3)
    ##
    ## 使用 Tensorly 进行 Tucker 分解
    ## 选择合适的秩组合
    #ranks = choose_ranks(conv_weights, 1.)
    #core, factors = tucker(conv_weights, rank=ranks)
    ##
    ## 重构原始张量
    #reconstructed_tensor = tucker_to_tensor((core, factors))
#
    ## core 和 factors 包含分解后的核心张量和因子矩阵
    #print("core.shape:", core.shape)
    #for i in range(len(factors)):
    #    print(f"factors[i].shape", factors[i].shape)
#
    ##比较 conv_weights 和 reconstructed_tensor
    #if not torch.allclose(conv_weights, reconstructed_tensor, atol=1e-6):
    #    difference = torch.abs(conv_weights - reconstructed_tensor)
    #    max_diff = torch.max(difference).item()
    #    mean_diff = torch.mean(difference).item()
    #    print(f"Max difference: {max_diff}, Mean difference: {mean_diff}")
    #else:
    #    print("Weights are identical.")
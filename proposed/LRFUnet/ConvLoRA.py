import torch
from torch import Tensor
import torch.nn as nn
import math
from typing import TypeVar, Union, Tuple

T = TypeVar('T')
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
_size_2_t = _scalar_or_tuple_2_t[int]

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

class ConvLoRA(nn.Conv2d, LoRALayer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: Union[str, _size_2_t] = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None,
                r=0, lora_alpha=1, lora_dropout=0., merge_weights=True):
        super(ConvLoRA, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * kernel_size[0], in_channels * kernel_size[0]))
            )
            self.lora_B = nn.Parameter(
              self.weight.new_zeros((out_channels//self.groups*kernel_size[0], r*kernel_size[0]))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.lora_reset_parameters()
        self.merged = False

    def lora_reset_parameters(self):
        self.reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B) 

    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.weight.data -= (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.weight.data += (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            return self._conv_forward(
                x, 
                self.weight + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling,
                self.bias
            )
        return self._conv_forward(x, self.weight, self.bias)
    
if __name__ == '__main__':
    feat = torch.randn(2, 64, 512, 512).cuda()
    conv1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1, dilation=1, bias=True).cuda()
    conv2 = ConvLoRA(64, 128, kernel_size=(3, 3), stride=1, padding=1, dilation=1, bias=True,
                    r=4).cuda()
    
    state_dict = conv1.state_dict()
    conv2.load_state_dict(state_dict, strict=False)

    out1 = conv1(feat)
    out2 = conv2(feat)
    #print(out1.shape)
    #print(out2.shape)

    if torch.allclose(out1, out2, atol=1e-6):
        print("out1 and out2 have the same values.")
    else:
        print("out1 and out2 have different values.")
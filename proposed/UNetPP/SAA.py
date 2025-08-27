import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math

class ds_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, 
                 dilation=[1,3,5], groups=1, bias=True, 
                 act_layer='nn.SiLU(True)', init='kaiming'):
        super().__init__()
        assert in_planes%groups==0
        assert kernel_size==3, 'only support kernel size 3 now'
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.with_bias = bias
        
        self.weight = nn.Parameter(torch.randn(out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_planes))
        else:
            self.bias = None
        self.act = eval(act_layer)
        self.init = init
        self._initialize_weights()

    def _initialize_weights(self):
        if self.init == 'dirac':
            nn.init.dirac_(self.weight, self.groups)
        elif self.init == 'kaiming':
            nn.init.kaiming_uniform_(self.weight)
        else:
            raise NotImplementedError
        if self.with_bias:
            if self.init == 'dirac':
                nn.init.constant_(self.bias, 0.)
            elif self.init == 'kaiming':
                bound = self.groups / (self.kernel_size**2 * self.in_planes)
                bound = math.sqrt(bound)
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                raise NotImplementedError

    def forward(self, x):
        output = 0
        for dil in self.dilation:
            output += self.act(
                F.conv2d(
                    x, weight=self.weight, bias=self.bias, stride=self.stride, padding=dil,
                    dilation=dil, groups=self.groups,
                )
            )
        return output

class Separable_Atrous_Attention(nn.Module):
    def __init__(self, dim, recursive=0, r_num=0, qkv_bias=True,):
        super().__init__()

        self.dim = dim
        self.recursive = recursive
        self.r_num = r_num

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.i = nn.Linear(dim, 1, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.out = nn.Linear(dim, dim)
        
        self.ds = ds_conv2d(
            dim, dim, kernel_size=3, stride=1, 
            dilation=[1, 3, 5], groups=dim, bias=qkv_bias, 
            act_layer='nn.SiLU(True)', init='kaiming',
        )
        if self.r_num > 1:
            self.silu = nn.SiLU(True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _inner_attention(self, x):
        x = x.permute(0, 2, 3, 1) #B, C, H, W -> B, H, W, C
        B, H, W, C = x.shape
        N = H * W
        q = self.q(x).permute(0, 3, 1, 2) #B, C, H, W
        q = self.ds(q).permute(0, 2, 3, 1) #B, H, W, C
        i = self.i(q).reshape(B, N, 1) #B, N, 1
        weight_i = torch.softmax(i, dim=1)

        k = self.k(x).reshape(B, N, C) #B, N, C
        v = self.v(x).reshape(B, N, C) #B, N, C

        context_score = weight_i * k #B, N, C
        #out = self.out(v * context_score).reshape(B, H, W, C)
        out = self.out(v * context_score).reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out
                
    def forward(self, x):
        if not self.recursive == 0:
            x_in = x
            x = self._inner_attention(x)
            if self.r_num > 1:
                x = self.silu(x)
            for _ in range(self.r_num-1):
                x = x + x_in
                x_in = x
                x = self._inner_attention(x)
                x = self.silu(x)
        else:
            x = self._inner_attention(x)
        return x
    
if __name__ == '__main__':
    feature = torch.randn(2, 512, 16, 16).cuda()
    attn = Separable_Atrous_Attention(dim=512).cuda()
    y = attn(feature)
    print(y.shape)
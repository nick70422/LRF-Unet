import numpy as np
import torch
import torch.nn as nn
from torchprofile import profile_macs
from nnunet.network_architecture.initialization import InitWeights_He

from UNetPP.generic_UNetPlusPlus import Generic_UNetPlusPlus
from UNetPP.generic_UNetPlusPlus_LRF import Generic_UNetPlusPlus_LRF as UNetPP_LRFenc
from nnUNet.generic_UNet import Generic_UNet
from SwinUnet.SwinUnet import SwinUnet
from SwinUnetV2.SwinUnetV2 import SwinUnetV2
from UNet3P.generic_UNet3P import Generic_UNet3P
#from SARUNet.models.se_p_resunet.se_p_resunet import Se_PPP_ResUNet
#from MMUNet import MMUNet
from LRFUnet.generic_MobileV3UNetPlusPlus_LRF import Generic_MobileV3UNetPlusPlus_LRF
#from LRFUnet.generic_MobileV4UNetPlusPlus_LRF import Generic_MobileV4UNetPlusPlus_LRF
from LRFUnet.generic_UNetPlusPlus_mobile_DW import Generic_MobileUNetPlusPlus_DW
#from lightweight_UNet3P.generic_UNet3P import Generic_MobileUNet3P
#from UNet.unet_model import UNet
from LRFUnet.generic_UNetPlusPlus_LRF import Generic_UNetPlusPlus_LRF as UNetPP_LRFonly

num_input_channels = 1
base_num_features = 32
num_classes = 3
net_num_pool_op_kernel_sizes = [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]
conv_per_stage = 2
conv_op = nn.Conv2d
dropout_op = nn.Dropout2d
norm_op = nn.InstanceNorm2d
norm_op_kwargs = {'eps': 1e-5, 'affine': True}
dropout_op_kwargs = {'p': 0, 'inplace': True}
net_nonlin = nn.LeakyReLU
net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
net_conv_kernel_sizes = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
use_LoRA = False
LoRA_location = 'QKVO'
supervised = False
SSL = False

#UNet++
def get_UNetPP():
    return Generic_UNetPlusPlus(num_input_channels, base_num_features, num_classes,
                                        len(net_num_pool_op_kernel_sizes),
                                        conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs,
                                        net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                        net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)

#UNet++ LRF AB
def get_UNetPP_LRF_AB():
    return UNetPP_LRFenc(num_input_channels, base_num_features, num_classes,
                                            len(net_num_pool_op_kernel_sizes),
                                            conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs,
                                            net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                            net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True,
                                            LRF_r=128, LRF_start_pos=0)

#UNet++ LRF 2Conv                     
def get_UNetPP_LRF_2Conv():
    return UNetPP_LRFenc(num_input_channels, base_num_features, num_classes,
                                            len(net_num_pool_op_kernel_sizes),
                                            conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs,
                                            net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                            net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True,
                                            LRF_type='2Conv', LRF_r=256, LRF_start_pos=0, use_CBAM=False, use_ASPP=False, use_attn='SAA')

#nnUNet
def get_nnUNet():
    return Generic_UNet(num_input_channels, base_num_features, num_classes,
                                    len(net_num_pool_op_kernel_sizes),
                                    conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)

#UNet+++
def get_UNet3P():
    return Generic_UNet3P(num_input_channels, base_num_features, num_classes,
                                    len(net_num_pool_op_kernel_sizes),
                                    conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)

#SwinUNet
def get_SwinUNet():
    return SwinUnet(3, 512, num_classes, conv_op, False)

#SwinUNetV2
def get_SwinUNetV2():
    return SwinUnetV2(3, 512, num_classes, conv_op, False, True, use_LoRA, LoRA_location, supervised, SSL)

#SARUNet
def get_SARUNet():
    return Se_PPP_ResUNet(1,3)

#MMUNet
def get_MMUNet():
    return MMUNet(in_channels=num_input_channels, num_classes=num_classes, base_channels= base_num_features, deep_supervision=False)

def get_MobileNetV3UNetPP_LRF_2Conv():
    return Generic_MobileV3UNetPlusPlus_LRF(num_input_channels, base_num_features, num_classes,
                                            len(net_num_pool_op_kernel_sizes),
                                            conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs,
                                            net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                            net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True,
                                            LRF_type='2Conv', LRF_r=256, LRF_start_pos=0, use_CBAM=False, use_ASPP=False)

def get_MobileNetV3UNetPP_LRF_AB():
    return Generic_MobileV3UNetPlusPlus_LRF(num_input_channels, base_num_features, num_classes,
                                            len(net_num_pool_op_kernel_sizes),
                                            conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs,
                                            net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                            net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True,
                                            LRF_type='AB', LRF_r=4, LRF_start_pos=0, use_CBAM=False, use_ASPP=False)

def get_MobileNetV4UNetPP_LRF_AB():
    return Generic_MobileV4UNetPlusPlus_LRF(num_input_channels, base_num_features, num_classes,
                                            len(net_num_pool_op_kernel_sizes),
                                            conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs,
                                            net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                            net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True,
                                            LRF_type='AB', LRF_r=128, LRF_start_pos=0, use_CBAM=False, use_ASPP=False)

def get_MobileNetV3UNetPP_DW():
    return Generic_MobileUNetPlusPlus_DW(num_input_channels, base_num_features, num_classes,
                                            len(net_num_pool_op_kernel_sizes),
                                            conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs,
                                            net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                            net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True,
                                            LRF_type='DW')

def get_lightweightUNet3P():
    return Generic_MobileUNet3P(num_input_channels, base_num_features, num_classes,
                                    len(net_num_pool_op_kernel_sizes),
                                    conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)

def get_UNet():
    return UNet(n_channels=num_input_channels, n_classes=num_classes)

def get_proposed_MV3only():
    return Generic_MobileV3UNetPlusPlus_LRF(num_input_channels, base_num_features, num_classes,
                                            len(net_num_pool_op_kernel_sizes),
                                            conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs,
                                            net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                            net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True,
                                            LRF_type='normal', LRF_r=128, LRF_start_pos=0, use_CBAM=False, use_ASPP=False)

def get_proposed_LRFCONVonly():
    return UNetPP_LRFonly(num_input_channels, base_num_features, num_classes,
                                            len(net_num_pool_op_kernel_sizes),
                                            conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs,
                                            net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                            net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True,
                                            LRF_type='AB', LRF_r=128)


if __name__ == '__main__':
    #model = get_nnUNet().cuda() #nnUNet
    #model = get_UNetPP().cuda() #UNet++
    #model = get_UNet3P().cuda() #UNet3+
    #model = get_SwinUNet().cuda() #SwinUNet
    #model = get_SwinUNetV2().cuda() #SwinV2UNet
    #model = get_SARUNet().cuda() #SARUNet
    model = get_MobileNetV3UNetPP_LRF_AB().cuda() #propsed
    #model = get_proposed_MV3only().cuda() #baseline+MV3 only
    #model = get_proposed_LRFCONVonly().cuda() #baseline+LRF-Conv only

    #未使用
    #model = get_UNetPP_LRF_AB().cuda()
    #model = get_UNetPP_LRF_2Conv().cuda()
    #model = get_MMUNet().cuda()
    #model = get_MobileNetV3UNetPP_LRF_2Conv().cuda()
    #model = get_MobileNetV3UNetPP_DW().cuda()
    #model = get_lightweightUNet3P().cuda()
    #model = get_UNet().cuda()


    model_parameters = filter(lambda p: p.requires_grad, (model).parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('No of model\'s params ', round(params / 1e6, 2))

    #model_parameters = filter(lambda p: p.requires_grad, (model.conv_blocks_context[:-1]).parameters())
    #params = sum([np.prod(p.size()) for p in model_parameters])
    #print('No of encoder params ', round(params / 1e6, 2))

    # 創建一個隨機輸入張量
    input_tensor = torch.randn(1, 1, 512, 512).cuda()

    # 計算FLOPs
    macs = profile_macs(model, input_tensor)
    flops = 2 * macs  # 每個MAC操作通常需要2個FLOP

    print(f"GFLOPs: {round(flops / 1e9, 2)}")
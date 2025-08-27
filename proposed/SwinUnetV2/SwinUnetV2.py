# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
try:
    from swin_unet_v2 import SwinTransformerSys
except:
    from .swin_unet_v2 import SwinTransformerSys
#from .swin_unet_v2_LoRA import SwinTransformerSys
#from swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from nnunet.network_architecture.neural_network import SegmentationNetwork

logger = logging.getLogger(__name__)

class SwinUnetV2(SegmentationNetwork):
    def __init__(self, num_input_channels, img_size=224, num_classes=21843, conv_op=False, resume=False, test=False, use_LoRA=False, LoRA_location='QKVO', supervised=False, SSL=False):
        super(SwinUnetV2, self).__init__()
        self.num_classes = num_classes
        self.conv_op = conv_op #this one is only for nnUNet framework
        #self.config = config

        self.swin_unet = SwinTransformerSys(img_size=img_size,
                                patch_size=4,
                                in_chans=num_input_channels,
                                num_classes=self.num_classes,
                                embed_dim=96,
                                depths=[2, 2, 6, 2],
                                num_heads=[3, 6, 12, 24],
                                window_size=int(img_size/32),
                                mlp_ratio=4,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False,
                                use_LoRA=use_LoRA,
                                LoRA_location=LoRA_location)
        if not resume and not test:
            if use_LoRA:
                self.my_load(supervised=supervised, SSL=SSL)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x)
        return logits

    def my_load(self, supervised=False, SSL=False):
        if SSL:
            pretrained_path = f'./pretrained_ckpt/mySSL_{SSL}.pth'
            #pretrained_path = '/mnt/d/Nick/medical/code/DL_models_benchmark-master/SwinUnetV2/pretrained_ckpt/my_SSL.pth'
            if pretrained_path is not None:
                print("pretrained_path:{}".format(pretrained_path))
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                #device = torch.device('cpu')
                pretrained_dict = torch.load(pretrained_path, map_location=device)['model']
                model_dict = self.swin_unet.state_dict()
                # 记录未加载预训练权重的模块
                not_loaded = []
                #freeze_module = []
    
                for k in model_dict.keys():
                    pretrain_k = f'encoder.{k}'
                    if pretrain_k not in pretrained_dict:
                        not_loaded.append(k)
                    elif pretrained_dict[pretrain_k].shape != model_dict[k].shape:
                        not_loaded.append(k)
                    elif 'up' in k or 'concat_back_dim' in k or 'output' in k:
                        not_loaded.append(k)
    
                # 加载匹配的权重
                for k, v in pretrained_dict.items():
                    pretrain_k = k[8:]
                    if pretrain_k in model_dict and v.shape == model_dict[pretrain_k].shape:
                        model_dict[pretrain_k] = v
    
                msg = self.swin_unet.load_state_dict(model_dict, strict=False)
    
                print("Modules without pre-trained weights:")
                for module in not_loaded:
                    print(module)
                # 冻结预训练模型的权重
                for name, param in self.swin_unet.named_parameters():
                    if name not in not_loaded:
                        #freeze_module.append(name)
                        param.requires_grad = False
            
                print("Pre-trained weights have been frozen")
                #print("Pre-trained weights have been frozen, listed below:")
                #for module in freeze_module:
                #    print(module)
        elif supervised:
            pretrained_path = './pretrained_ckpt/mendeley3class_3000best.model'
            #pretrained_path = '/mnt/d/Nick/medical/code/DL_models_benchmark-master/SwinUnetV2/pretrained_ckpt/mendeley3class_3000best.model'
            if pretrained_path is not None:
                print("pretrained_path:{}".format(pretrained_path))
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                #device = torch.device('cpu')
                pretrained_dict = torch.load(pretrained_path, map_location=device)['state_dict']
                model_dict = self.state_dict()
                # 记录未加载预训练权重的模块
                not_loaded = []

                for k in model_dict.keys():
                    if k not in pretrained_dict:
                        not_loaded.append(k)
                    elif pretrained_dict[k].shape != model_dict[k].shape:
                        not_loaded.append(k)
                    elif 'up' in k or 'concat_back_dim' in k or 'output' in k:
                        not_loaded.append(k)

                # 加载匹配的权重
                for k, v in pretrained_dict.items():
                    if k in model_dict and v.shape == model_dict[k].shape:
                        model_dict[k] = v

                msg = self.load_state_dict(model_dict, strict=False)

                print("Modules without pre-trained weights:")
                for module in not_loaded:
                    print(module)
                # 冻结预训练模型的权重
                for name, param in self.named_parameters():
                    if name not in not_loaded:
                        param.requires_grad = False

                print("Pre-trained weights have been frozen.")
        
    
    def load_from(self, img_size):
        if img_size == 224:
            #pretrained_path = "./pretrained_ckpt/swin_tiny_patch4_window7_224.pth"
            pretrained_path = "/mnt/d/Nick/medical/code/DL_models_benchmark-master/SwinUnetV2/pretrained_ckpt/swin_tiny_patch4_window7_224.pth"
            if pretrained_path is not None:
                print("pretrained_path:{}".format(pretrained_path))
                device = torch.device('cpu')
                #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                pretrained_dict = torch.load(pretrained_path, map_location=device)
                if "model"  not in pretrained_dict:
                    print("---start load pretrained modle by splitting---")
                    pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                    for k in list(pretrained_dict.keys()):
                        if "output" in k:
                            print("delete key:{}".format(k))
                            del pretrained_dict[k]
                    msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                    print(msg)
                    return
                pretrained_dict = pretrained_dict['model']
                print("---start load pretrained modle of swin encoder---")

                model_dict = self.swin_unet.state_dict()
                full_dict = copy.deepcopy(pretrained_dict)
                for k, v in pretrained_dict.items():
                    if "layers." in k:
                        current_layer_num = 3-int(k[7:8])
                        current_k = "layers_up." + str(current_layer_num) + k[8:]
                        full_dict.update({current_k:v})
                for k in list(full_dict.keys()):
                    if k in model_dict:
                        if full_dict[k].shape != model_dict[k].shape:
                            print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                            del full_dict[k]

                msg = self.swin_unet.load_state_dict(full_dict, strict=False)
                print(msg)
            else:
                print("none pretrain")
        elif img_size == 512:
            pretrained_path = './pretrained_ckpt/mendeley3class_3000best.model'
            #pretrained_path = 'SwinUnetV2/pretrained_ckpt/mendeley3class_3000best.model'
            if pretrained_path is not None:
                print("pretrained_path:{}".format(pretrained_path))
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                pretrained_dict = torch.load(pretrained_path, map_location=device)
                pretrained_dict = pretrained_dict['state_dict']
                print("---start load pretrained modle of swin encoder---")

                msg = self.load_state_dict(pretrained_dict)
                print(msg)
            else:
                print("none pretrain")
 
if __name__ == '__main__':
    image = torch.randn(1, 1, 512, 512)
    #model = SwinUnetV2(3, 512, 3, use_LoRA=True, LoRA_location='QKVO')
    #model = SwinUnetV2(3, 512, 3, use_LoRA=True, LoRA_location='QKVO', SSL=True)
    model = SwinUnetV2(3, 512, 3, use_LoRA=True, LoRA_location='QKVO', supervised=True)
    #model = SwinUnetV2(3, 224, 3)
    #model = SwinUnetV2(3, 512, 3)
    
    out = model(image)
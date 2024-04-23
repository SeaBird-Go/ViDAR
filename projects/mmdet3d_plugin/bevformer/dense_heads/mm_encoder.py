'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-02-07 09:43:01
Email: haimingzhang@link.cuhk.edu.cn
Description: Multi-modal encoder.
'''

import torch
from torch import nn
import copy
from mmengine.registry import MODELS
from mmengine.model import BaseModule
import numpy as np

from .mimo_modules.Layers import (
    EncoderLayer, DecoderLayer, CrossAttnLayer)


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@MODELS.register_module()
class MMEncoder(nn.Module):
    def __init__(self, 
                 in_channel, 
                 img_in_channel, 
                 model_channel, 
                 N, 
                 heads,
                 use_sensor_type_embed=False):
        super().__init__()
        self.N = N
        self.model_channel = model_channel
        self.layers = get_clones(EncoderLayer(model_channel, heads), N)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, model_channel, 3, 2, 1, bias=False),
            nn.GroupNorm(1, model_channel),
            nn.SiLU(inplace=True),
        )  # for occupancy map
        
        self.conv_img = nn.Sequential(
            nn.Conv2d(img_in_channel, model_channel, 3, 2, 1, bias=False),
            nn.GroupNorm(1, model_channel),
            nn.SiLU(inplace=True),
        )

        if use_sensor_type_embed:
            self.sensor_type_embedding = nn.Parameter(torch.randn((2, 128)))

    def forward(self, x, img, src_pos, 
                src_ego_pose=None, 
                use_attn_mask=False):
        bs, seq, c, h, w = x.size()
        
        ## for occupancy map
        x = self.conv(x.view(-1, *x.shape[2:]))
        x = x.view(bs, seq, self.model_channel, x.shape[-2], x.shape[-1])
        x = x + src_pos

        ## for image
        img = self.conv_img(img.view(-1, *img.shape[2:]))
        img = img.view(bs, seq, self.model_channel, img.shape[-2], img.shape[-1])
        img = img + src_pos

        if src_ego_pose is not None:
            x = x + src_ego_pose
            img = img + src_ego_pose
        
        if hasattr(self, 'sensor_type_embedding'):
            bs, F, C, H, W = x.shape
            x_emb = self.sensor_type_embedding[0].reshape(1, 1, C, 1, 1).expand(bs, F, C, H, W)
            img_emb = self.sensor_type_embedding[1].reshape(1, 1, C, 1, 1).expand(bs, F, C, H, W)

            x = x + x_emb
            img = img + img_emb
        
        # combine the occupancy map and image
        x = torch.cat([x, img], dim=1)  # concatenate along the sequence length dimension

        attn_mask = None
        if use_attn_mask:
            fusion_seq_len = x.shape[1]
            attn_mask = torch.zeros((fusion_seq_len, fusion_seq_len), dtype=torch.float32).to(x.device)
            attn_mask[:seq, -seq:] = 1.0
            attn_mask = attn_mask * -10000.0

        for i in range(self.N):
            x = self.layers[i](x, attn_mask)
        
        x_encoded = x[:, :seq]
        img_encoded = x[:, seq:]
        return x_encoded, img_encoded
    

@MODELS.register_module()
class MMCrossAttnEncoder(nn.Module):
    """Fuse the multi-modal inputs with cross attention mechanism.

    Args:
        nn (_type_): _description_
    """
    def __init__(self, 
                 in_channel, 
                 img_in_channel, 
                 model_channel, 
                 N, 
                 heads):
        super().__init__()
        self.N = N
        self.model_channel = model_channel
        self.layers = get_clones(CrossAttnLayer(model_channel, heads), N)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, model_channel, 3, 2, 1, bias=False),
            nn.GroupNorm(1, model_channel),
            nn.SiLU(inplace=True),
        )  # for occupancy map
        
        self.conv_img = nn.Sequential(
            nn.Conv2d(img_in_channel, model_channel, 3, 2, 1, bias=False),
            nn.GroupNorm(1, model_channel),
            nn.SiLU(inplace=True),
        )

    def forward(self, x, img, src_pos, src_ego_pose=None):
        bs, seq, c, h, w = x.size()
        
        ## for occupancy map
        x = self.conv(x.view(-1, *x.shape[2:]))
        x = x.view(bs, seq, self.model_channel, x.shape[-2], x.shape[-1])
        x = x + src_pos

        ## for image
        img = self.conv_img(img.view(-1, *img.shape[2:]))
        img = img.view(bs, seq, self.model_channel, img.shape[-2], img.shape[-1])
        img = img + src_pos

        if src_ego_pose is not None:
            x = x + src_ego_pose
            img = img + src_ego_pose
        
        for i in range(self.N):
            x = self.layers[i](x, img)
        
        return x, None
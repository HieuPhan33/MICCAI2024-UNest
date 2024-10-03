# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np
import os, sys

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# import kornia
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import models.vit_seg_configs as configs
from models.vit_seg_modeling_resnet_skip import ResNetV2


logger = logging.getLogger(__name__)



def unpatchify(features):
    B, N, C = features.size()
    H = W = int(np.sqrt(N))
    x = features.transpose(1, 2).reshape(B, C, H, W)
    return x

def patchify(features):
    x = features.flatten(2).transpose(1, 2)
    return x

# def morphological_gradient(binary_seg, n=3, threshold=0.5):
#     binary_seg = torch.where(binary_seg > threshold, torch.tensor(1.0, device=binary_seg.device), torch.tensor(0.0, device=binary_seg.device))
#     # Define the structuring element (a 3x3 square in this case)
#     structuring_element = torch.ones((n, n), device=binary_seg.device)

#     # Apply dilation
#     dilation = kornia.morphology.dilation(binary_seg, structuring_element)

#     # Apply erosion
#     erosion = kornia.morphology.erosion(binary_seg, structuring_element)


#     # Calculate the morphological gradient
#     gradient = dilation - erosion

#     # Normalize the gradient to the range [0, 1]
#     gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-10)
#     return gradient

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class HybridAttention(nn.Module):
    def __init__(self, config, window_size, vis=False, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = config.hidden_size
        self.vis=vis
        self.num_heads = config.transformer["num_heads"]
        self.attention_head_dim = int(config.hidden_size / self.num_heads)
        self.all_head_size = self.num_heads * self.attention_head_dim

        self.window_size = window_size  # Wh, Ww
        self.scale = qk_scale or self.attention_head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), self.num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.query_g = Linear(config.hidden_size, self.all_head_size, bias=qkv_bias)
        self.key_g = Linear(config.hidden_size, self.all_head_size, bias=qkv_bias)
        self.value_g = Linear(config.hidden_size, self.all_head_size, bias=qkv_bias)

        self.query_l = Linear(config.hidden_size, self.all_head_size, bias=qkv_bias)
        self.key_l = Linear(config.hidden_size, self.all_head_size, bias=qkv_bias)
        self.value_l = Linear(config.hidden_size, self.all_head_size, bias=qkv_bias)

        self.attn_drop = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.proj_g = nn.Linear(self.dim, self.dim)
        self.proj_l = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(config.transformer["attention_dropout_rate"])

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
         
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)


    def sparse_attention(self, q, k ,v, mask, mask_kv=None):
        if mask_kv is None:
            mask_kv = mask
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
            
        attention_scores = attention_scores / math.sqrt(self.attention_head_dim)
        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(-1) & mask_kv.unsqueeze(1).unsqueeze(-2)
            attention_scores = attention_scores.masked_fill(attn_mask == 0, float('-1e9'))
        attention_probs = self.softmax(attention_scores)
        #attention_probs[torch.isnan(attention_probs)] = 0.
        #attention_probs = torch.nan_to_num(attention_probs)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_drop(attention_probs)

        context_layer = torch.matmul(attention_probs, v)
        return context_layer, weights
    
    def global_forward(self, hidden_states, mask=None, edge_mask=None):
        B, N, C = hidden_states.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # mixed_query_layer, mixed_key_layer, mixed_value_layer = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        mixed_query_layer = self.query_g(hidden_states)
        mixed_key_layer = self.key_g(hidden_states)
        mixed_value_layer = self.value_g(hidden_states)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # context_layer = torch.matmul(attention_probs, value_layer)
        context_layer, weights = self.sparse_attention(query_layer, key_layer, value_layer, mask=mask, mask_kv=edge_mask)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if mask is not None:
            context_layer = context_layer[mask]
        # if mask is not None:
        #     assert context_layer[~mask].sum() == 0, "Masked positions need to have 0 outputs"
        attention_output = self.proj_g(context_layer)
        attention_output = self.proj_drop(attention_output)
        return attention_output, weights

    
    def local_forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape # (B*n_windows, window_area, C)
        
        q = self.query_l(x)
        k = self.key_l(x)
        v = self.value_l(x)
        
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)
        
        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj_l(x)
        x = self.proj_drop(x)
        return x, None
    
    def forward(self, x, mask=None, edge_mask=None, mode='local'):
        if mode == 'local':
            return self.local_forward(x, mask)
        elif mode == 'global':
            return self.global_forward(x, mask, edge_mask=edge_mask)


class SplitAttention(nn.Module):
    def __init__(self, config, vis=False, qkv_bias=True):
        super(SplitAttention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size, bias=qkv_bias)
        self.key = Linear(config.hidden_size, self.all_head_size, bias=qkv_bias)
        self.value = Linear(config.hidden_size, self.all_head_size, bias=qkv_bias)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        attn_mask = mask.unsqueeze(1).unsqueeze(-1) & mask.unsqueeze(1).unsqueeze(-2) # B, 1, L, L

        fg_attention_scores = attention_scores.masked_fill(attn_mask == 0, float('-1e9'))
        fg_attention_probs = self.softmax(fg_attention_scores)
        
        bg_attention_scores = attention_scores.masked_fill(attn_mask == 1, float('-1e9'))
        bg_attention_probs = self.softmax(bg_attention_scores) # B, H, L, L
        
        attention_probs = torch.zeros(fg_attention_probs.shape).to(fg_attention_probs.device)
        fg_mask = (attn_mask==1).repeat(1, self.num_attention_heads, 1, 1)
        bg_mask = ~fg_mask
        attention_probs[fg_mask] = fg_attention_probs[fg_mask]
        attention_probs[bg_mask] = bg_attention_probs[bg_mask]
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights



class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

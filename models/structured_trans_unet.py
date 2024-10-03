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
import torch.nn.functional as F
from models.attention_parts import *
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import models.vit_seg_configs as configs
from models.vit_seg_modeling_resnet_skip import ResNetV2


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False
        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.binary_seg = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                Conv2dReLU(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    padding=1,
                    use_batchnorm=True,
                ),
                nn.UpsamplingBilinear2d(scale_factor=2),
                Conv2dReLU(
                    in_channels,
                    in_channels,
                    kernel_size=1,
                    use_batchnorm=True,
                ),
                nn.Conv2d(in_channels, 1, 1),
                nn.Sigmoid()
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        H, W = x.size(-2), x.size(-1)
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        original = x.clone()
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        
        ### Binary seg
        original = original.detach()
        seg = self.binary_seg(original)
        return embeddings, features, seg

        
    

class StructuredBlock(nn.Module):
    def __init__(self, config, vis, input_resolution=(16,16), drop_path=0.):
        super(StructuredBlock, self).__init__()
        self.hidden_size = config.hidden_size
        self.window_size = config.transformer.window_size
        self.input_resolution = input_resolution
        self.fth = config.transformer.fth
        # self.local_attn = WindowAttention(
        #     config.hidden_size, window_size=to_2tuple(self.window_size), num_heads=config.transformer["num_heads"],
        #     qkv_bias=True)
        self.attn = HybridAttention(config,
            window_size=to_2tuple(self.window_size), qkv_bias=True)
        
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        #self.attn = Attention(config, vis)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.edge_attention = config.transformer.edge_attention

    def forward(self, x, mask=None, edge=None):
        H, W = self.input_resolution
        shortcut = x
        x = self.attention_norm(x)
        B, L, C = x.size()
        #combined_features = torch.zeros(x.size())
        if mask is not None:
            mask = F.avg_pool2d(mask, kernel_size=4, stride=4).to(x.device)
            mask = mask.permute(0, 2, 3, 1).squeeze(-1)

            mask = (mask > self.fth).int()

            edge_mask = None
            if self.edge_attention:
                edge_mask = F.max_pool2d(edge, kernel_size=4, stride=4).to(x.device)
                edge_mask = edge_mask.squeeze(1)
                edge_mask = edge_mask > 0.5
                edge_mask = edge_mask.view(B, L)
        
            fg_mask = mask == 1
            fg_mask = fg_mask.view(B, L)
            bg_mask = (mask == 0).float()
            bg_mask_shaped = bg_mask.unsqueeze(1)
            bg_mask_windowed = F.avg_pool2d(bg_mask_shaped, kernel_size=self.window_size, stride=self.window_size).view(-1) # Debug avg -> max in real
            bg_mask_windowed = (bg_mask_windowed > 0.0).bool() # B x n_windows
            '''
            x: B, L, C
            FG
            fg = x[mask_fg]
            x_fg = SelfAttention(fg)
            x_new[mask_fg] = x_fg
            
            BG:
            x = window_partition(x) -- (B * n_win) * win_size * win_size * C
            masked_win = avg_pool(mask_bg, win_size).flatten() -- B * n_win
            x_bg = x[masked_win] -- n_bg_win * win_size * win_size * C
            x_bg = LocalAttention(x_bg)
            x_new[masked_win] = x_bg
            
            
            '''
            x_shaped = x.view(B, H, W, C)
            ### Local attention in background
            x_windows = window_partition(x_shaped, self.window_size)

            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
            x_windows_bg = x_windows[bg_mask_windowed]
            # W-MSA/SW-MSA
            attn_windows, _ = self.attn(x_windows_bg, mask=None, mode='local')  # nW_bg*B, window_size*window_size, C
            x_new = torch.zeros(x_windows.shape).to(attn_windows.device) # nW * B, window_size*window_size, C
            x_new[bg_mask_windowed] = attn_windows

            # merge windows
            x_new = x_new.view(-1, self.window_size, self.window_size, C)

            # reverse cyclic shift
            x_new = window_reverse(x_new, self.window_size, H, W)  # B H' W' C
            x_new = x_new.view(B, L, C)


            ### Shape-aware attention in foreground
            #weights = None
            x_fg, weights = self.attn(x, mask=fg_mask.view(B, L), edge_mask=edge_mask, mode='global')
            x_new[fg_mask] = x_fg
        else:
            x_new, weights = self.attn(x, mode='global')
        # x = shortcut + self.drop_path(x_new)
        # x = x + self.drop_path(self.ffn(self.ffn_norm(x)))
        x = shortcut + x_new
        x = x + self.ffn(self.ffn_norm(x))
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query_g.weight.copy_(query_weight)
            self.attn.key_g.weight.copy_(key_weight)
            self.attn.value_g.weight.copy_(value_weight)
            self.attn.query_l.weight.copy_(query_weight)
            self.attn.key_l.weight.copy_(key_weight)
            self.attn.value_l.weight.copy_(value_weight)
            self.attn.proj_g.weight.copy_(out_weight)
            self.attn.proj_l.weight.copy_(out_weight)
            
            self.attn.query_g.bias.copy_(query_bias)
            self.attn.key_g.bias.copy_(key_bias)
            self.attn.value_g.bias.copy_(value_bias)
            self.attn.proj_g.bias.copy_(out_bias)
            self.attn.query_l.bias.copy_(query_bias)
            self.attn.key_l.bias.copy_(key_bias)
            self.attn.value_l.bias.copy_(value_bias)
            self.attn.proj_l.bias.copy_(out_bias)


            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis, input_resolution):
        super(Encoder, self).__init__()
        self.vis = vis
        self.edge_attention = config.transformer.edge_attention
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.conv_trans = config.conv_trans
        if config.conv_trans:
            block = ConvTransBlock
        else:
            block = StructuredBlock
        for _ in range(config.transformer["num_layers"]):
            layer = block(config, vis, input_resolution=input_resolution)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states, binary_seg=None):
        if binary_seg is not None and self.edge_attention:
            edge_mask = morphological_gradient(binary_seg, n=3)
            edge_mask = edge_mask.detach()
        else:
            edge_mask = None
        attn_weights = []
        injection = unpatchify(hidden_states)
        for layer_block in self.layer:
            if self.conv_trans:
                hidden_states, injection, weights = layer_block(hidden_states, injection=injection, mask=binary_seg, edge=edge_mask)
            else:
                hidden_states, weights = layer_block(hidden_states, mask=binary_seg, edge=edge_mask)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights





class StructureTransformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(StructureTransformer, self).__init__()
        self.img_size = img_size
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis, input_resolution=(img_size//16, img_size//16))
        self.shape_assist = False

    def forward(self, input_ids, binary_seg=None):
        embedding_output, features, seg = self.embeddings(input_ids)
        # if binary_seg is not None:
        #     if self.shape_assist:
        #         binary_seg = torch.zeros(seg.shape)
        #         binary_seg[seg > 0.5] = 1.0
        #         binary_seg = binary_seg.detach()
        #     else:
        #         binary_seg = None
        if binary_seg is None:
            binary_seg = torch.zeros(seg.shape)
            binary_seg[seg > 0.5] = 1.0
        else:
            binary_seg = F.interpolate(binary_seg, size=(seg.shape[-2], seg.shape[-1]))
        binary_seg = binary_seg.detach()
        encoded, attn_weights = self.encoder(embedding_output, binary_seg)  # (B, n_patch, hidden)
        return encoded, attn_weights, features, seg


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class TranslationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        #upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = nn.Tanh()
        super().__init__(conv2d, activation)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class StructuredVisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(StructuredVisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.img_size = img_size
        self.transformer = StructureTransformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.head = TranslationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config
    
    def toggle_shape_assist(self):
        self.transformer.shape_assist = not self.transformer.shape_assist

    def forward(self, x, seg=None):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, attn_weights, features, seg = self.transformer(x, seg)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.head(x)
        return logits, F.interpolate(seg, size=(self.img_size, self.img_size), mode='bilinear')

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                #if self.classifier == "seg":
                _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'SF-B_16': configs.get_sf_b16_config(),
    'testing': configs.get_testing(),
}


if __name__ == '__main__':
    torch.manual_seed(0)
    import random; random.seed(0)
    import time, os
    torch.use_deterministic_algorithms(True)
    x = torch.rand(4,1,256,256)
    seg = (torch.rand(4,1,256,256) > 0.2).float()
    print(seg.float().mean())
    config = CONFIGS['SF-B_16']
    config.conv_trans = True
    config.transformer.num_layers = 3
    config.transformer.edge_attention = False
    net = StructuredVisionTransformer(config, img_size=256, num_classes=21843, zero_head=False, vis=False)
    net.load_from(weights = np.load(config.pretrained_path))
    net(x, seg)
    # net.load_from(weights = np.load(config.pretrained_path))
    # for _ in range(10):
    #     net(x, seg)
    
    # values = []
    # for _ in range(10):
    #     starttime = time.time()
    #     out, seg = net(x, seg)
    #     diff = time.time() - starttime
    #     values.append(diff)
    # print(np.mean(values))
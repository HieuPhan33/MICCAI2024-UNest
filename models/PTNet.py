# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
Borrow from timm(https://github.com/rwightman/pytorch-image-models)
"""
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath
import math
from timm.models.registry import register_model


class PTNet(nn.Module):
    """
    PTNet class
    """

    def __init__(self, img_size=[224, 256], trans_type='performer', down_ratio=[1, 1, 2, 4], channels=[1, 32, 64, 128],
                 patch=[7, 3, 3], embed_dim=512, depth=12, num_heads=4, mlp_ratio=2., drop_rate=0.,
                 qkv_bias=False, qk_scale=None, attn_drop_rate=0., drop_path_rate=0, norm_layer=nn.LayerNorm,
                 skip_connection=True, individual_use=True):
        super().__init__()
        self.individual = individual_use
        self.down_blocks = nn.ModuleList(
            [SWT4(img_size=[int(img_size[0] / down_ratio[i]), int(img_size[1] / down_ratio[i])], in_channel=channels[i],
                  out_channel=channels[i + 1], patch=patch[i],
                  stride=int(down_ratio[i + 1] / down_ratio[i]),
                  padding=int((patch[i] - 1) / 2)) for i in range(len(down_ratio) - 1)])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.bottleneck = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=nn.LeakyReLU(negative_slope=0.2))
            for i in range(depth)])
        if not skip_connection:
            self.up_blocks = nn.ModuleList(
                [SWT_up4(img_size=[int(img_size[0] / down_ratio[-(i + 1)]), int(img_size[1] / down_ratio[-(i + 1)])],
                         in_channel=channels[-(i + 1)],
                         out_channel=channels[-(i + 1) - 1], patch=patch[-(i + 1)],
                         up_scale=int(down_ratio[-(i + 1)] / down_ratio[-(i + 1) - 1]),
                         padding=int((patch[-(i + 1)] - 1) / 2)) for i in range(1, len(down_ratio) - 2)])
            self.up_blocks.append(SWT_up4(img_size=[int(img_size[0] / down_ratio[1]), int(img_size[1] / down_ratio[1])],
                                          in_channel=channels[1],
                                          out_channel=channels[1], patch=patch[0],
                                          up_scale=int(down_ratio[1] / down_ratio[0]),
                                          padding=int((patch[0] - 1) / 2)))
        else:

            self.up_blocks = (nn.ModuleList(
                [SWT_up4(img_size=[int(img_size[0] / down_ratio[-(i + 1)]), int(img_size[1] / down_ratio[-(i + 1)])],
                         in_channel=2 * channels[-(i + 1)],
                         out_channel=channels[-(i + 1) - 1], patch=patch[-(i + 1)],
                         up_scale=int(down_ratio[-(i + 1)] / down_ratio[-(i + 1) - 1]),
                         padding=int((patch[-(i + 1)] - 1) / 2)) for i in range(1, len(down_ratio) - 2)]))
            self.up_blocks.append(SWT_up4(img_size=[int(img_size[0] / down_ratio[1]), int(img_size[1] / down_ratio[1])],
                                          in_channel=2 * channels[1],
                                          out_channel=channels[1], patch=patch[0],
                                          up_scale=int(down_ratio[2] / down_ratio[1]),
                                          padding=int((patch[0] - 1) / 2)))
            self.up_blocks.append(SWT_up4(img_size=img_size, in_channel=channels[1] + 1,
                                          out_channel=channels[1], patch=patch[0],
                                          up_scale=int(down_ratio[1] / down_ratio[0]),
                                          padding=int((patch[0] - 1) / 2)))
        # self.sw = nn.Unfold(kernel_size=1, stride=1, padding=0)
        self.PE = nn.Parameter(data=get_sinusoid_encoding(
            n_position=(img_size[0] // down_ratio[-1]) * (img_size[1] // down_ratio[-1]),
            d_hid=embed_dim), requires_grad=False)

        self.final_proj = nn.Linear(channels[1], 1)
        # self.final_LN = nn.LayerNorm(channels[1])

        self.bot_proj = nn.Linear(channels[-2] * patch[-1] * patch[-1], embed_dim)
        # self.bot_LN = nn.LayerNorm(channels[-1])
        self.bot_proj2 = nn.Linear(embed_dim, channels[-2])
        # self.bot_LN2 = nn.LayerNorm(embed_dim)
        self.tanh = nn.Tanh()
        self.sc = skip_connection
        self.size = img_size
        self.ratio = down_ratio

    def forward(self, x):
        x0 = x
        if not self.sc:
            for i, down in enumerate(self.down_blocks[:-1]):
                x = down(x)
                B, HW, C = x.shape
                x = x.transpose(1, 2).reshape(B, C, int(self.size[0] / self.ratio[i + 1]),
                                              int(self.size[1] / self.ratio[i + 1]))

            x = self.down_blocks[-1](x)
            x = self.bot_proj(x)
            x = x + self.PE
            for blk in self.bottleneck:
                x = blk(x)
            x = self.bot_proj2(x).transpose(1, 2)
            B, C, HW = x.shape
            x = x.reshape(B, C, int(self.size[0] / self.ratio[-1]), int(self.size[1] / self.ratio[-1]))
            for up in self.up_blocks[:-1]:
                x = up(x, size=[x.shape[2], x.shape[3]])
            x = self.up_blocks[-1](x, reshape=False)
        else:
            SC = []
            for i, down in enumerate(self.down_blocks[:-1]):
                # print(x.shape)
                x = down(x)
                # print(x.shape)
                B, HW, C = x.shape
                x = x.transpose(1, 2).reshape(B, C, int(self.size[0] / self.ratio[i + 1]),
                                              int(self.size[1] / self.ratio[i + 1]))
                SC.append(x)
            x = self.down_blocks[-1](x, attention=False)
            x = self.bot_proj(x)
            x = x + self.PE
            for blk in self.bottleneck:
                x = blk(x)
            x = self.bot_proj2(x).transpose(1, 2)
            B, C, HW = x.shape
            x = x.reshape(B, C, int(self.size[0] / self.ratio[-1]), int(self.size[1] / self.ratio[-1]))
            for i, up in enumerate(self.up_blocks[:-1]):
                x = up(x, size=[x.shape[2], x.shape[3]], SC=SC[-(i + 1)], reshape=True)
            if not self.individual:
                x = self.up_blocks[-1](x, SC=x0, reshape=True, size=[x.shape[2], x.shape[3]])
            else:
                x = self.up_blocks[-1](x, SC=x0, reshape=False)

        if not self.individual:
            return x
        else:
            x = self.final_proj(x).transpose(1, 2)
            B, C, HW = x.shape
            x = x.reshape(B, C, self.size[0], self.size[1])
            return self.tanh(x)


class PTNet_local(nn.Module):
    def __init__(self, img_size=[224, 256], trans_type='performer', down_ratio=[1, 1, 2, 4, 8],
                 channels=[1, 32, 64, 128, 256],
                 patch=[7, 3, 3, 3, 3], embed_dim=256, depth=9, num_heads=4, mlp_ratio=2., drop_rate=0.,
                 qkv_bias=False, qk_scale=None, attn_drop_rate=0., drop_path_rate=0, norm_layer=nn.LayerNorm,
                 skip_connection=True):
        super().__init__()
        self.GlobalGenerator = PTNet(img_size=[int(img_size[0] / 2), int(img_size[1] / 2)],
                                   down_ratio=[1, 1, 2, 4, 8],
                                   channels=[1, 32, 64, 128, 256],
                                   patch=[7, 3, 3, 3, 3], embed_dim=512, depth=9, individual_use=False,
                                   skip_connection=True)
        self.down_blocks = nn.ModuleList(
            [SWT4(img_size=[int(img_size[0] / down_ratio[i]), int(img_size[1] / down_ratio[i])], in_channel=channels[i],
                  out_channel=channels[i + 1], patch=patch[i],
                  stride=int(down_ratio[i + 1] / down_ratio[i]),
                  padding=int((patch[i] - 1) / 2)) for i in range(len(down_ratio) - 1)])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.bottleneck = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=nn.LeakyReLU(negative_slope=0.2))
            for i in range(depth)])
        if not skip_connection:
            self.up_blocks = nn.ModuleList(
                [SWT_up4(img_size=[int(img_size[0] / down_ratio[-(i + 1)]), int(img_size[1] / down_ratio[-(i + 1)])],
                         in_channel=channels[-(i + 1)],
                         out_channel=channels[-(i + 1) - 1], patch=patch[-(i + 1)],
                         up_scale=int(down_ratio[-(i + 1)] / down_ratio[-(i + 1) - 1]),
                         padding=int((patch[-(i + 1)] - 1) / 2)) for i in range(1, len(down_ratio) - 2)])
            self.up_blocks.append(SWT_up4(img_size=[int(img_size[0] / down_ratio[1]), int(img_size[1] / down_ratio[1])],
                                          in_channel=channels[1],
                                          out_channel=channels[1], patch=patch[0],
                                          up_scale=int(down_ratio[1] / down_ratio[0]),
                                          padding=int((patch[0] - 1) / 2)))
        else:

            self.up_blocks = (nn.ModuleList(
                [SWT_up4(img_size=[int(img_size[0] / down_ratio[-(i + 1)]), int(img_size[1] / down_ratio[-(i + 1)])],
                         in_channel=2 * channels[-(i + 1)],
                         out_channel=channels[-(i + 1) - 1], patch=patch[-(i + 1)],
                         up_scale=int(down_ratio[-(i + 1)] / down_ratio[-(i + 1) - 1]),
                         padding=int((patch[-(i + 1)] - 1) / 2)) for i in range(1, len(down_ratio) - 2)]))
            self.up_blocks.append(SWT_up4(img_size=[int(img_size[0] / down_ratio[1]), int(img_size[1] / down_ratio[1])],
                                          in_channel=2 * channels[1] + 32,
                                          out_channel=channels[1], patch=patch[0],
                                          up_scale=int(down_ratio[2] / down_ratio[1]),
                                          padding=int((patch[0] - 1) / 2)))
            self.up_blocks.append(SWT_up4(img_size=img_size, in_channel=channels[1] + 1,
                                          out_channel=channels[1], patch=patch[0],
                                          up_scale=int(down_ratio[1] / down_ratio[0]),
                                          padding=int((patch[0] - 1) / 2)))
        # self.sw = nn.Unfold(kernel_size=1, stride=1, padding=0)
        self.PE = nn.Parameter(data=get_sinusoid_encoding(
            n_position=(img_size[0] // down_ratio[-1]) * (img_size[1] // down_ratio[-1]),
            d_hid=embed_dim), requires_grad=False)

        self.final_proj = nn.Linear(channels[1], 1)
        # self.final_LN = nn.LayerNorm(channels[1])

        self.bot_proj = nn.Linear(channels[-2] * patch[-1] * patch[-1], embed_dim)
        # self.bot_LN = nn.LayerNorm(channels[-1])
        self.bot_proj2 = nn.Linear(embed_dim, channels[-2])
        # self.bot_LN2 = nn.LayerNorm(embed_dim)
        self.tanh = nn.Tanh()
        self.sc = skip_connection
        self.size = img_size
        self.ratio = down_ratio

    def forward(self, x):
        x0 = x
        Global_res = self.GlobalGenerator(F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True))
        if not self.sc:
            for i, down in enumerate(self.down_blocks[:-1]):
                x = down(x)
                B, HW, C = x.shape
                x = x.transpose(1, 2).reshape(B, C, int(self.size[0] / self.ratio[i + 1]),
                                              int(self.size[1] / self.ratio[i + 1]))

            x = self.down_blocks[-1](x)
            x = self.bot_proj(x)
            x = x + self.PE
            for blk in self.bottleneck:
                x = blk(x)
            x = self.bot_proj2(x).transpose(1, 2)
            B, C, HW = x.shape
            x = x.reshape(B, C, int(self.size[0] / self.ratio[-1]), int(self.size[1] / self.ratio[-1]))
            for up in self.up_blocks[:-1]:
                x = up(x, size=[x.shape[2], x.shape[3]])
            x = self.up_blocks[-1](x, reshape=False)
        else:
            SC = []
            for i, down in enumerate(self.down_blocks[:-1]):
                x = down(x)
                B, HW, C = x.shape
                x = x.transpose(1, 2).reshape(B, C, int(self.size[0] / self.ratio[i + 1]),
                                              int(self.size[1] / self.ratio[i + 1]))
                SC.append(x)
            x = self.down_blocks[-1](x, attention=False)
            x = self.bot_proj(x)
            x = x + self.PE
            for blk in self.bottleneck:
                x = blk(x)
            x = self.bot_proj2(x).transpose(1, 2)
            B, C, HW = x.shape
            x = x.reshape(B, C, int(self.size[0] / self.ratio[-1]), int(self.size[1] / self.ratio[-1]))
            for i, up in enumerate(self.up_blocks[:-2]):
                x = up(x, SC=SC[-(i + 1)], reshape=True, size=[x.shape[2], x.shape[3]])
            for up in self.up_blocks[-2:-1]:
                x = up(torch.cat((x, Global_res), dim=1), SC=SC[0], reshape=True, size=[x.shape[2], x.shape[3]])

            x = self.up_blocks[-1](x, SC=x0, reshape=False)

        x = self.final_proj(x).transpose(1, 2)
        B, C, HW = x.shape
        x = x.reshape(B, C, self.size[0], self.size[1])
        return self.tanh(x)


@register_model
def PTN(img_size=[224, 256], **kwargs):
    model = PTNet(img_size=img_size, down_ratio=[1, 1, 2, 4, 8, 16],
                channels=[1, 32, 64, 128, 256, 512],
                patch=[7, 3, 3, 3, 3], embed_dim=512, depth=3, individual_use=True)

    return model


@register_model
def PTN_local(img_size=[224, 256], **kwargs):
    model = PTNet_local(img_size=img_size, **kwargs)

    return model


class Token_performer(nn.Module):
    def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0., dp2 = 0.):
        super().__init__()
        self.emb = in_dim * head_cnt # we use 1, so it is no need here
        self.kqv = nn.Linear(dim, 3 * self.emb)
        self.dp = nn.Dropout(dp1)
        self.proj = nn.Linear(self.emb, self.emb)
        self.head_cnt = head_cnt
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(self.emb)
        self.epsilon = 1e-8  # for stable in division

        self.mlp = nn.Sequential(
            nn.Linear(self.emb, 1 * self.emb),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.GELU(),
            nn.Linear(1 * self.emb, self.emb),
            nn.Dropout(dp2),
        )

        self.m = int(self.emb * kernel_ratio)
        self.w = torch.randn(self.m, self.emb)
        self.w = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(self.m), requires_grad=False)

    def prm_exp(self, x):
        # part of the function is borrow from https://github.com/lucidrains/performer-pytorch
        # and Simo Ryu (https://github.com/cloneofsimo)
        # ==== positive random features for gaussian kernels ====
        # x = (B, T, hs)
        # w = (m, hs)
        # return : x : B, T, m
        # SM(x, y) = E_w[exp(w^T x - |x|/2) exp(w^T y - |y|/2)]
        # therefore return exp(w^Tx - |x|/2)/sqrt(m)
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m) / 2
        wtx = torch.einsum('bti,mi->btm', x.float(), self.w)

        return torch.exp(wtx - xd) / math.sqrt(self.m)

    def single_attn(self, x):
        k, q, v = torch.split(self.kqv(x), self.emb, dim=-1)
        kp, qp = self.prm_exp(k), self.prm_exp(q)  # (B, T, m), (B, T, m)
        D = torch.einsum('bti,bi->bt', qp, kp.sum(dim=1)).unsqueeze(dim=2)  # (B, T, m) * (B, m) -> (B, T, 1)
        kptv = torch.einsum('bin,bim->bnm', v.float(), kp)  # (B, emb, m)
        y = torch.einsum('bti,bni->btn', qp, kptv) / (D.repeat(1, 1, self.emb) + self.epsilon)  # (B, T, emb)/Diag
        # skip connection
        y = v + self.dp(self.proj(y))  # same as token_transformer in T2T layer, use v as skip connection

        return y

    def forward(self, x):
        x = self.single_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.LeakyReLU(negative_slope=0.2), drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention2(nn.Module):
    def __init__(self, dim, num_heads=8, in_dim = None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, in_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B,C,N = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)


        x = (attn @ v).transpose(1, 2).reshape(B, N, self.in_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        # skip connection
        x = v.squeeze(1) + x   # because the original x has different size with current x, use v to do skip connection

        return x

class Token_transformer(nn.Module):

    def __init__(self, dim, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            in_dim,  num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_dim)
        self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=in_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        print(x.shape)
        x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x













class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU(), norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class SWT(nn.Module):
    """
    sliding window tokenization
    """

    def __init__(self, img_size,in_channel=1, out_channel=64, patch=4, stride=2, padding=1):
        super().__init__()
        self.sw = nn.Unfold(kernel_size=patch, stride=stride, padding=padding)
        num_patches = (img_size // stride) * (img_size // stride)
        self.transformer = Token_performer(dim=out_channel, in_dim=out_channel)
        self.proj = nn.Linear(in_channel*patch*patch,out_channel)
        self.PE = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches, d_hid=out_channel),
                               requires_grad=True)
        self.LN = nn.LayerNorm(out_channel)
    def forward(self, x):
        x = self.sw(x).transpose(1, 2)
        x = self.proj(x)
        x = self.LN(x) + self.PE

        x = self.transformer(x)


        return x

class SWT_up(nn.Module):
    """
    sliding window tokenization
    """

    def __init__(self, up_scale, img_size, in_channel=1, out_channel=64, patch=4, stride=1, padding=1):
        super().__init__()
        self.sw = nn.Unfold(kernel_size=patch, stride=stride, padding=padding)
        num_patches = (img_size // stride) * (img_size // stride)

        self.transformer = Token_performer(dim=in_channel*patch*patch, in_dim=out_channel)

        #self.PE = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches, d_hid=in_channel*patch*patch))
        self.up_sample = nn.UpsamplingBilinear2d(scale_factor=up_scale)

    def forward(self, x):

        x = self.sw(x).transpose(1, 2)
        x = self.transformer(x)
        B, HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(HW)), int(np.sqrt(HW)))

        return self.up_sample(x)

class SWT2(nn.Module):
    """
    sliding window tokenization
    """

    def __init__(self, img_size,in_channel=1, out_channel=64, patch=4, stride=2, padding=1):
        super().__init__()
        self.sw = nn.Unfold(kernel_size=patch, stride=stride, padding=padding)
        num_patches = (img_size // stride) * (img_size // stride)
        self.transformer = Token_performer(dim=in_channel*patch*patch, in_dim=out_channel)
        self.transformer2 = Token_performer(dim=out_channel,in_dim=out_channel)
    def forward(self, x, attention=True):
        x = self.sw(x).transpose(1, 2)
        if attention:
            x = self.transformer(x)
            x = self.transformer2(x)

        return x

class SWT_up2(nn.Module):
    """
    sliding window tokenization
    """

    def __init__(self, up_scale, img_size, in_channel=1, out_channel=64, patch=4, stride=1, padding=1):
        super().__init__()
        self.sw = nn.Unfold(kernel_size=patch, stride=stride, padding=padding)
        num_patches = (img_size // stride) * (img_size // stride)

        self.transformer = Token_performer(dim=in_channel, in_dim=out_channel)
        self.transformer2 = Token_performer(dim=out_channel, in_dim=out_channel)

        #self.PE = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches, d_hid=in_channel*patch*patch))
        self.up_sample = nn.UpsamplingBilinear2d(scale_factor=up_scale)

    def forward(self, x, SC=None, reshape=True):
        x = self.up_sample(x)
        if SC is not None:
            x = torch.cat((x,SC), dim=1)
        x = x.flatten(start_dim=2).transpose(1, 2)
        x = self.transformer(x)
        x = self.transformer2(x)
        if reshape:
            B, HW, C = x.shape
            x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(HW)), int(np.sqrt(HW)))
        return x


class SWT3(nn.Module):
    """
    sliding window tokenization
    """

    def __init__(self, img_size,in_channel=1, out_channel=64, patch=4, stride=2, padding=1):
        super().__init__()
        self.sw = nn.Unfold(kernel_size=patch, stride=stride, padding=padding)
        num_patches = (img_size // stride) * (img_size // stride)
        self.transformer = Token_performer(dim=out_channel, in_dim=out_channel)
        self.proj = nn.Linear(in_channel*patch*patch,out_channel)
        self.transformer2 = Token_performer(dim=out_channel,in_dim=out_channel)
        self.LN = nn.LayerNorm(out_channel)
        self.PE = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches, d_hid=out_channel),requires_grad=False)
    def forward(self, x, attention=True):
        x = self.sw(x).transpose(1, 2)
        if attention:
            x = self.proj(x)
            x = self.LN(x) + self.PE
            x = self.transformer(x)
            x = self.transformer2(x)

        return x

class SWT_up3(nn.Module):
    """
    sliding window tokenization
    """

    def __init__(self, up_scale, img_size, in_channel=1, out_channel=64, patch=4, stride=1, padding=1):
        super().__init__()
        self.sw = nn.Unfold(kernel_size=patch, stride=stride, padding=padding)
        # num_patches = (img_size // stride) * (img_size // stride)

        self.transformer = Token_performer(dim=in_channel, in_dim=out_channel)
        #self.transformer2 = Token_performer(dim=out_channel, in_dim=out_channel)
        # self.proj = nn.Linear(in_channel,out_channel)
        # self.LN = nn.LayerNorm(out_channel)
        # self.PE = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches, d_hid=out_channel),requires_grad=False)
        self.up_sample = nn.UpsamplingBilinear2d(scale_factor=up_scale)

    def forward(self, x, SC=None, reshape=True):
        x = self.up_sample(x)
        if SC is not None:
            x = torch.cat((x,SC), dim=1)
        x = x.flatten(start_dim=2).transpose(1, 2)
        # x = self.proj(x)
        # x = self.LN(x) + self.PE
        x = self.transformer(x)
        #x = self.transformer2(x)
        if reshape:
            B, HW, C = x.shape
            x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(HW)), int(np.sqrt(HW)))
        return x


class SWT4(nn.Module):
    """
    sliding window tokenization
    """

    def __init__(self, img_size,in_channel=1, out_channel=64, patch=4, stride=2, padding=1):
        super().__init__()
        self.sw = nn.Unfold(kernel_size=patch, stride=stride, padding=padding)
        self.transformer = Token_performer(dim=in_channel*patch*patch, in_dim=out_channel)

    def forward(self, x, attention=True):
        x = self.sw(x).transpose(1, 2)
        if attention:

            x = self.transformer(x)

        return x

class SWT_up4(nn.Module):
    """
    sliding window tokenization
    """

    def __init__(self, up_scale, img_size, in_channel=1, out_channel=64, patch=4, stride=1, padding=1):
        super().__init__()
        self.sw = nn.Unfold(kernel_size=patch, stride=stride, padding=padding)
        # num_patches = (img_size // stride) * (img_size // stride)

        self.transformer = Token_performer(dim=in_channel, in_dim=out_channel)

        self.up_sample = nn.UpsamplingBilinear2d(scale_factor=up_scale)
        self.factor = up_scale
    def forward(self, x, size=None, SC=None, reshape=True):
        x = self.up_sample(x)
        if SC is not None:
            x = torch.cat((x,SC), dim=1)
        x = x.flatten(start_dim=2).transpose(1, 2)

        x = self.transformer(x)
        if reshape:
            B, HW, C = x.shape
            x = x.transpose(1, 2).reshape(B, C, int(size[0]*self.factor), int(size[1]*self.factor))
        return x
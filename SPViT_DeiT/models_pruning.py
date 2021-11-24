# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# Modifications copyright (c) 2021 Zhuang AI Group, Haoyu He

import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
from params import args
import numpy as np
import torch.nn.functional as F
from torch.distributions.utils import clamp_probs


__all__ = [
    'spvit_deit_tiny_patch16_224', 'spvit_deit_small_patch16_224',
    'spvit_deit_base_patch16_224',
]


def relaxed_bernoulli_logits(probs, temperature):
    probs = clamp_probs(probs)
    uniforms = clamp_probs(torch.rand(probs.shape, dtype=probs.dtype, device=probs.device))
    return (uniforms.log() - (-uniforms).log1p() + probs.log() - (-probs).log1p()) / temperature


def bernoulli_sample(probs, temperature=1.0):
    logits = relaxed_bernoulli_logits(probs, temperature)
    y_soft = torch.sigmoid(logits)
    y_hard = (logits > 0.0).float()
    ret = y_hard.detach() - y_soft.detach() + y_soft
    return ret


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class UnifiedMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., theta=0., ffn_indicators=None):
        super().__init__()

        self.in_features = in_features
        out_features = out_features or in_features
        self.hidden_features = hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

        if ffn_indicators is None:

            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)

            # Threshold parameters
            self.register_parameter('ffn_thresholds', nn.Parameter(torch.tensor([theta] * hidden_features)))

            # The indicators
            self.register_buffer('assigned_indicator_index', nn.Parameter(torch.zeros(hidden_features)))
            self.fine_tuning = False
        else:
            self.fc1 = nn.Linear(in_features, ffn_indicators.nonzero().shape[0])
            self.fc2 = nn.Linear(ffn_indicators.nonzero().shape[0], out_features)

            self.fine_tuning = True

    def forward(self, x):
        if not self.fine_tuning:
            return self.search_forward(x)
        else:
            return self.finetune_forward(x)

    def search_forward(self, x):

        ffn_probs = F.sigmoid(self.ffn_thresholds)
        if self.training:
            ffn_indicators = bernoulli_sample(ffn_probs)
        else:
            ffn_indicators = (ffn_probs > 0.5).float()

        x = self.fc1(x)
        x = self.act(x)

        x = ffn_indicators.unsqueeze(0).unsqueeze(0) * x
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        # We derive the FFN indicators by expectation, and
        # ffn_indicators are kept to calculate complexity loss
        self.ffn_indicators = (ffn_probs > 0.5).float() - torch.sigmoid(
            ffn_probs - 0.5).detach() + torch.sigmoid(ffn_probs - 0.5)

        self.register_buffer('assigned_indicator_index', self.ffn_indicators)
        return x, torch.sum(self.ffn_indicators)

    def finetune_forward(self, x):

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x, None


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0., locality_strength=None, use_local_init=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class UnifiedAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0., window_size=None, alpha=1e-2, theta=0.0, msa_indicators=None):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.window_size = window_size

        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.max_kernel_size = 3
        self.msa_indicators = msa_indicators
        self.alpha = alpha

        if msa_indicators is None:

            msa_type = 'search'

            threshold_num = 3

            # We separate qk from v here
            self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.v = nn.Linear(dim, dim, bias=qkv_bias)

            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)

            # Get the pre-defined convolution regions for deriving convolutional layer outputs
            rel_indices_pad = self.get_rel_indices((self.window_size[0] + 2) * (self.window_size[1] + 2))

            self.att_mask_3x3 = self.conv_attmask_init(3, rel_indices=rel_indices_pad)
            conv_3x3_idx_tmp = (self.att_mask_3x3 > 1e-5).nonzero()
            conv_3x3_idx_list = []
            for idx in range((self.window_size[0] + 2) * (self.window_size[1] + 2)):
                token_idxes = torch.where(conv_3x3_idx_tmp[:, 2] == idx)[0]
                token_idxes = torch.stack([conv_3x3_idx_tmp[token_idxes][:, 3], conv_3x3_idx_tmp[token_idxes][:, 1]],
                                          dim=1)
                conv_3x3_idx_list.append(token_idxes)
            self.conv_3x3_idx = torch.stack(conv_3x3_idx_list, dim=0).view(-1, 2)

            self.bn_1x1 = nn.BatchNorm2d(head_dim)
            self.bn_3x3 = nn.BatchNorm2d(head_dim)
            self.conv_act = nn.ReLU()

            self.register_parameter('kernel_thresholds', nn.Parameter(torch.tensor([theta] * threshold_num)))
            self.register_parameter("head_probs", nn.Parameter(torch.ones([self.num_heads, (self.max_kernel_size ** 2)])))
        elif not msa_indicators[0]:

            msa_type = 'skip-connection'
        elif not msa_indicators[1]:

            msa_type = '1x1 Bconv'
            self.bn_1x1 = nn.BatchNorm2d(head_dim)
            self.register_parameter("head_probs", nn.Parameter(torch.ones([self.num_heads, (self.max_kernel_size ** 2)])))
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.conv_act = nn.ReLU()

        elif not msa_indicators[2]:

            msa_type = '3x3 Bconv'
            self.bn_3x3 = nn.BatchNorm2d(head_dim)
            self.register_parameter("head_probs", nn.Parameter(torch.ones([self.num_heads, (self.max_kernel_size ** 2)])))
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.conv_act = nn.ReLU()

        elif msa_indicators[2]:

            msa_type = 'msa'
            self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.v = nn.Linear(dim, dim, bias=qkv_bias)

            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)

        else:
            raise NotImplementedError

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def conv_attmask_init(self, kernel_size=1, rel_indices=None):
        # Modified from https://github.com/facebookresearch/convit, thanks for the great work!
        center = (kernel_size - 1) / 2 if kernel_size % 2 == 0 else kernel_size // 2

        pos_proj0 = torch.zeros(kernel_size ** 2, 3).to(rel_indices.device)
        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                position = h1 + kernel_size * h2
                pos_proj0[position, 2] = -1
                pos_proj0[position, 1] = 2 * (h1 - center) * 1
                pos_proj0[position, 0] = 2 * (h2 - center) * 1

        # Get the same order as the value/projection matrix
        pos_proj = pos_proj0

        pos_proj *= 46
        rel_indices = rel_indices
        attmask = torch.matmul(rel_indices, pos_proj.transpose(1, 0)).permute(0, 3, 1, 2)

        return attmask.softmax(dim=-1)

    def forward(self, x):
        if self.msa_indicators is None:
            return self.search_forward(x)
        else:
            return self.finetune_forward(x)

    def search_forward(self, x):

        indicators = self.single_path_init()

        B, N, C = x.shape
        N_patches = N - 1

        v_weight = F.linear(x, self.v.weight)
        v = (v_weight + self.v.bias).reshape(B, N, -1, C // self.num_heads).permute(0, 2, 1, 3)
        attn = self.get_attention(x)

        msa_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        msa_out = self.proj(msa_out)
        msa_out = self.proj_drop(msa_out)

        cls_token = x[:, 0].unsqueeze(1)

        head_probs = (self.head_probs.view(self.num_heads, (self.max_kernel_size ** 2)) / self.alpha).softmax(0)

        # Scaling the attention heads
        new_v_weight = (v_weight[:, 1:].view(B, N_patches, self.num_heads, self.head_dim).permute(0, 1, 3, 2) @ head_probs).permute(0, 1, 3, 2) # B_ * N * num_heads * head_dim
        new_v_bias = (self.v.bias.view(self.num_heads, self.head_dim).permute(1, 0) @ head_probs) # head_dim * num_heads
        new_proj = (self.proj.weight.view(self.dim, self.num_heads, self.head_dim).permute(0, 2, 1) @ head_probs) # dim * head_dim * num_heads

        H = W = int(N_patches ** 0.5)

        v_weight_1x1 = new_v_weight[:, :, 4]
        v_bias_1x1 = new_v_bias[:, 4]
        proj_weight_1x1 = new_proj[..., 4]

        conv_1x1_out = v_weight_1x1 + v_bias_1x1
        conv_1x1_out = self.conv_act(self.bn_1x1(
            conv_1x1_out.view(B, H, W, self.head_dim).permute(0, 3, 1, 2))).permute(0, 2, 3, 1)
        conv_1x1_out = F.linear(conv_1x1_out.reshape(B, N_patches, -1), proj_weight_1x1)

        conv_3x3_out = new_v_weight[:, :, :9]
        # Indexing to get conv outputs
        conv_3x3_out = F.pad(conv_3x3_out.view(B, H, W, 9, self.head_dim), pad=(0, 0, 0, 0, 1, 1, 1, 1))
        conv_3x3_out = (conv_3x3_out.flatten(1, 2)[:, self.conv_3x3_idx[:, 0], self.conv_3x3_idx[:, 1]].view(B, self.window_size[0] + 2, self.window_size[1] + 2, 9, self.head_dim)[:, 1:-1, 1:-1].sum(3) + \
                       new_v_bias[:, :9].sum(dim=-1)).permute(0, 3, 1, 2)
        conv_3x3_out = self.conv_act(self.bn_3x3(conv_3x3_out)).permute(0, 2, 3, 1).flatten(1, 2)
        conv_3x3_out = F.linear(conv_3x3_out, new_proj[..., :9].sum(-1))

        conv_1x1_out = torch.cat([cls_token, conv_1x1_out + self.proj.bias], dim=1)
        conv_3x3_out = torch.cat([cls_token, conv_3x3_out + self.proj.bias], dim=1)

        outputs = indicators[2] * msa_out + \
                  indicators[1] * (1 - indicators[2]) * conv_3x3_out + \
                  indicators[0] * torch.prod(1 - indicators[1:], dim=0) * conv_1x1_out

        kernel_thresholds = F.sigmoid(self.kernel_thresholds)
        kernel_indicators = (kernel_thresholds > 0.5).float() - torch.sigmoid(kernel_thresholds - 0.5).detach() + torch.sigmoid(kernel_thresholds - 0.5)

        # kernel_indicators are kept to calculate complexity loss
        self.kernel_indicators = torch.cumprod(kernel_indicators, dim=0)

        return outputs, self.kernel_indicators.detach(), kernel_thresholds.detach()

    def finetune_forward(self, x):

        B, N, C = x.shape
        indicators = self.msa_indicators

        cls_token = x[:, 0].unsqueeze(1)
        x_nocls = x[:, 1:]

        if not indicators[0]:
            x = x
        elif not indicators[1]:

            head_probs = (self.head_probs.view(self.num_heads, (self.max_kernel_size ** 2)) / self.alpha).softmax(0)

            new_v_weight = self.v.weight.view(self.num_heads, self.head_dim, self.dim).permute(1, 2, 0) @ head_probs
            new_v_bias = self.v.bias.view(self.num_heads, self.head_dim).permute(1, 0) @ head_probs
            new_proj_weight = self.proj.weight.view(self.dim, self.num_heads, self.head_dim).permute(0, 2, 1) @ head_probs

            x_nocls = F.linear(x_nocls, new_v_weight[..., 4], new_v_bias[..., 4])
            x_nocls = x_nocls.view(x_nocls.shape[0], int(x_nocls.shape[1] ** 0.5), int(x_nocls.shape[1] ** 0.5),
                                   x_nocls.shape[2]).permute(0, 3, 1, 2)
            x_nocls = self.conv_act(self.bn_1x1(x_nocls)).permute(0, 2, 3, 1).flatten(1, 2)
            x_nocls = F.linear(x_nocls, new_proj_weight[..., 4], self.proj.bias)

            x = torch.cat([cls_token, x_nocls], dim=1)

        elif not indicators[2]:

            head_probs = (self.head_probs.view(self.num_heads, (self.max_kernel_size ** 2)) / self.alpha).softmax(0)

            new_v_weight = self.v.weight.view(self.num_heads, self.head_dim, self.dim).permute(1, 2, 0) @ head_probs
            new_v_bias = (self.v.bias.view(self.num_heads, self.head_dim).permute(1, 0) @ head_probs).sum(-1)
            new_proj_weight = (self.proj.weight.view(self.dim, self.num_heads, self.head_dim).permute(0, 2, 1) @ head_probs).sum(-1)
            kernel_3x3 = new_v_weight.permute(2, 0, 1).view(3, 3, self.head_dim, -1).permute(2, 3, 1, 0)

            # The bias for conv is the bias sum of nine heads
            x_nocls = x_nocls.view(x_nocls.shape[0], int(x_nocls.shape[1] ** 0.5), int(x_nocls.shape[1] ** 0.5), x_nocls.shape[2]).permute(0, 3, 1, 2)
            x_nocls = F.conv2d(x_nocls, kernel_3x3, padding=1, bias=new_v_bias)
            x_nocls = self.conv_act(self.bn_3x3(x_nocls)).permute(0, 2, 3, 1).flatten(1, 2)
            x_nocls = F.linear(x_nocls, new_proj_weight, self.proj.bias)

            x = torch.cat([cls_token, x_nocls], dim=1)

        else:
            v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            attn = self.get_attention(x)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)

        return x, None, None

    def single_path_init(self):
        kernel_probs = F.sigmoid(self.kernel_thresholds)

        if self.training:
            kernel_indicators = bernoulli_sample(kernel_probs)
        else:
            kernel_indicators = (kernel_probs > 0.5).float()

        kernel_indicators = torch.cumprod(kernel_indicators, dim=0)

        # Slow down the gradient for less-complex gates
        if kernel_indicators[0] and kernel_indicators[1]:
            kernel_indicators[0] = kernel_indicators[0].detach()
            if kernel_indicators[2]:
                kernel_indicators[1] = kernel_indicators[1].detach()

        return kernel_indicators

    def get_attention(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        return attn

    def assign_indicators(self, indicators):
        self.assigned_indicators = self.kernel_indicators = indicators

    def get_rel_indices(self, num_patches):
        img_size = int(num_patches ** .5)
        rel_indices = torch.zeros(1, num_patches, num_patches, 3)
        ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size, img_size)
        indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
        indd = indx ** 2 + indy ** 2
        rel_indices[:, :, :, 2] = indd.unsqueeze(0)
        rel_indices[:, :, :, 1] = indy.unsqueeze(0)
        rel_indices[:, :, :, 0] = indx.unsqueeze(0)
        return rel_indices.cuda()


def get_attention_flops(module, input, output):

    input = input[0]
    B, N, C = input.shape
    # q @ k and attn @ v
    module.__flops__ += np.prod(input.shape) * N * 2


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=0.0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PruningBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, att_layer=None, ffn_layer=None,
                 window_size=None, theta=0., msa_indicators=None, ffn_indicators=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.norm1 = norm_layer(dim)
        self.attn = att_layer(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop, window_size=window_size, theta=theta, msa_indicators=msa_indicators)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_hidden_dim = mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                             drop=drop, theta=theta, ffn_indicators=ffn_indicators)

    def forward(self, x):

        shortcut = x
        x, kernel_indicators, kernel_thresholds = self.attn(self.norm1(x))

        x = shortcut + self.drop_path(x)

        ffn_shortcut = x
        x, ffn_indicators = self.mlp(self.norm2(x))

        x = ffn_shortcut + self.drop_path(x)
        return x, kernel_indicators, kernel_thresholds, ffn_indicators


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.embed_dim = embed_dim
        self.in_chans = in_chans

        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.apply(self._init_weights)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])

        return flops


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


# Original DeiT
class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, block=Block, ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x, [None], [None]


class SPVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, att_layer=None, ffn_layer=None,
                 loss_lambda=None, target_flops=None, theta=0., msa_indicators=None, ffn_indicators=None):
        super().__init__()

        att_layer = eval(att_layer)
        ffn_layer = eval(ffn_layer)

        self.loss_lambda = loss_lambda
        self.target_flops = target_flops

        # We use assigned_indicators as the fine-tune flag
        self.assigned_indicators = msa_indicators

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if self.assigned_indicators:
            self.blocks = nn.ModuleList([
                PruningBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, att_layer=att_layer,
                    ffn_layer=ffn_layer, window_size=(img_size//patch_size, img_size//patch_size), theta=theta,
                    msa_indicators=msa_indicators[i], ffn_indicators=ffn_indicators[i])
                for i in range(depth)])

        else:
            self.blocks = nn.ModuleList([
                PruningBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, att_layer=att_layer,
                    ffn_layer=ffn_layer, window_size=(img_size//patch_size, img_size//patch_size), theta=theta)
                for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        kernel_indicators_list = []
        kernel_thresholds_list = []
        ffn_indicators_list = []

        for blk in self.blocks:
            x, kernel_indicators, kernel_thresholds, ffn_indicators = blk(x)
            kernel_indicators_list.append(kernel_indicators)
            kernel_thresholds_list.append(kernel_thresholds)
            ffn_indicators_list.append(ffn_indicators)

        x = self.norm(x)
        return x[:, 0], kernel_indicators_list, kernel_thresholds_list, ffn_indicators_list

    def forward(self, x):
        x, kernel_indicators_list, kernel_thresholds_list, ffn_indicators_list = self.forward_features(x)
        x = self.head(x)
        return x, kernel_indicators_list, kernel_thresholds_list, ffn_indicators_list

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        flops += self.num_features * self.num_classes

        H, W = self.patch_embed.patches_resolution
        layer_flops_list = []

        att_flops = 0
        ffn_flops = 0

        for block in self.blocks:

            conv_1x1_flops = (1 * block.dim * (block.dim // block.num_heads) + (block.dim // block.num_heads) * block.dim) * (H * W)
            conv_3x3_flops = (3 * 3 * block.dim * (block.dim // block.num_heads) + \
                              (block.dim // block.num_heads) * block.dim) * (H * W)
            msa_flops = 4 * (H * W + 1) * block.dim**2 + 2 * block.dim * (H * W + 1)**2

            # Attention
            block_indicators = block.attn.kernel_indicators
            conv_1x1_flops = torch.tensor(conv_1x1_flops).to(block_indicators[0].device) if isinstance(conv_1x1_flops,
                                                                          int) else conv_1x1_flops
            conv_3x3_flops = torch.tensor(conv_3x3_flops).to(block_indicators[0].device) if isinstance(conv_3x3_flops,
                                                                                                       int) else conv_3x3_flops
            msa_flops = torch.tensor(msa_flops).to(block_indicators[0].device) if isinstance(msa_flops,
                                                                                             int) else msa_flops
            block_att = block_indicators[0] * conv_1x1_flops + \
                           block_indicators[1] * (conv_3x3_flops - conv_1x1_flops) + \
                           block_indicators[2] * (msa_flops - conv_3x3_flops)
            att_flops += block_att
            layer_flops_list.append(block_att / 1e9)

            # MLP
            ffn_indicators = block.mlp.ffn_indicators
            sum_ffn_indicators = torch.sum(ffn_indicators)
            block_mlp = 2 * H * W * block.dim * sum_ffn_indicators
            ffn_flops += block_mlp

            # Norms
            flops += block.dim * (H * W + 1) * 2
            flops += block_att + block_mlp

        return flops, att_flops, ffn_flops

    def calculate_bops_loss(self):

        flops, att_flops, ffn_flops = self.flops()

        bops = (torch.tensor(self.target_flops).to(flops.device) - flops / 1e9)**2

        bops_loss = bops * self.loss_lambda

        return bops_loss


@register_model
def spvit_deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = SPVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def spvit_deit_small_patch16_224(pretrained=False, **kwargs):
    model = SPVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def spvit_deit_base_patch16_224(pretrained=False, **kwargs):
    model = SPVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model
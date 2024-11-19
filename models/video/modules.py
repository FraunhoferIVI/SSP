import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from torch.autograd import Variable, Function


# https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/mmaction/models/backbones/swin_transformer.py
class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0) # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ShiftedWindows_MSA_3D(nn.Module):
    def __init__(self, embed_dim, window_size, shift_size, n_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(ShiftedWindows_MSA_3D, self).__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.n_heads = n_heads
        self.shift_size = shift_size

        self.attn = WindowAttention3D(
            embed_dim, window_size=window_size, num_heads=n_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)

    def forward(self, x):
        B, T, H, W, C = x.shape
        window_size, shift_size = self.get_window_size((T, H, W), self.window_size, self.shift_size)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - T % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Tp, Hp, Wp, _ = x.shape
        mask_matrix = self.compute_mask(Tp, Hp, Wp, window_size, shift_size, x.device)
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = self.window_partition(shifted_x, window_size) # B*nW, Wd*Wh*Ww, C

        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C

        t, h, w = window_size
        attn_windows = attn_windows.view(-1, t, h, w, C)
        shifted_x = self.window_reverse(attn_windows, window_size, B, Tp, Hp, Wp)  # B D' H' W' C

        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 >0 or pad_r > 0 or pad_b > 0:
            x = x[:, :T, :H, :W, :].contiguous()
        return x

    def window_partition(self, x, window_size):
        t, h, w = window_size
        B, T, H, W, C = x.shape
        x = x.view(B, T//t, t, H//h, h, W//w, w, C)
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, t*h*w, C)
        return windows
    
    def window_reverse(self, windows, window_size, B, T, H, W):
        t, h, w = window_size
        x = windows.view(B, T//t, H//h, W//w, t, w, h, -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, T, H, W, -1)
        return x
    
    def get_window_size(self, x_size, window_size, shift_size=None):
        use_window_size = list(window_size)
        if shift_size is not None:
            use_shift_size = list(shift_size)
        for i in range(len(x_size)):
            if x_size[i] <= window_size[i]:
                use_window_size[i] = x_size[i]
                if shift_size is not None:
                    use_shift_size[i] = 0

        if shift_size is None:
            return tuple(use_window_size)
        else:
            return tuple(use_window_size), tuple(use_shift_size)

    def compute_mask(self, T, H, W, window_size, shift_size, device):
        img_mask = torch.zeros((1, T, H, W, 1), device=device)  # 1 Dp Hp Wp 1
        cnt = 0
        for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0],None):
            for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1],None):
                for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2],None):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1
        mask_windows = self.window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
        mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

#https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/backbones/mix_transformer.py
class Mix_FFN3D(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.):
        super(Mix_FFN3D, self).__init__()
        self.mlp_1 = nn.Linear(embed_dim, hidden_dim)
        self.depth_conv = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.gelu = nn.GELU()
        self.mlp_2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.mlp_1(x)
        x = x.permute(0,4,1,2,3) # B C T H W
        x = self.depth_conv(x)
        x = x.permute(0,2,3,4,1) # B T H W C
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.mlp_2(x)
        x = self.dropout(x)
        return x


# https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/mmaction/models/backbones/swin_transformer.py
class WindowAttention2D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1

        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wh*Ww, Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0) # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ShiftedWindows_MSA_2D(nn.Module):
    def __init__(self, embed_dim, window_size, shift_size, n_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(ShiftedWindows_MSA_2D, self).__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.n_heads = n_heads
        self.shift_size = shift_size

        self.attn = WindowAttention2D(
            embed_dim, window_size=window_size, num_heads=n_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        window_size, shift_size = self.get_window_size((H, W), self.window_size, self.shift_size)
        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        mask_matrix = self.compute_mask(Hp, Wp, window_size, shift_size, x.device)
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = self.window_partition(shifted_x, window_size) # B*nW, Wh*Ww, C

        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wh*Ww, C

        h, w = window_size
        attn_windows = attn_windows.view(-1, h, w, C)
        shifted_x = self.window_reverse(attn_windows, window_size, B, Hp, Wp)  # B H' W' C

        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        return x

    def window_partition(self, x, window_size):
        h, w = window_size
        B, H, W, C = x.shape
        x = x.view(B, H//h, h, W//w, w, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, h*w, C)
        return windows
    
    def window_reverse(self, windows, window_size, B, H, W):
        h, w = window_size
        x = windows.view(B, H//h, W//w, w, h, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x
    
    def get_window_size(self, x_size, window_size, shift_size=None):
        use_window_size = list(window_size)
        if shift_size is not None:
            use_shift_size = list(shift_size)
        for i in range(len(x_size)):
            if x_size[i] <= window_size[i]:
                use_window_size[i] = x_size[i]
                if shift_size is not None:
                    use_shift_size[i] = 0

        if shift_size is None:
            return tuple(use_window_size)
        else:
            return tuple(use_window_size), tuple(use_shift_size)

    def compute_mask(self, H, W, window_size, shift_size, device):
        img_mask = torch.zeros((1, H, W, 1), device=device)  # 1 Hp Wp 1
        cnt = 0
        for h in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0],None):
            for w in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1],None):
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = self.window_partition(img_mask, window_size)  # nW, ws[0]*ws[1], 1
        mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

#https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/backbones/mix_transformer.py
class Mix_FFN2D(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.):
        super(Mix_FFN2D, self).__init__()
        self.mlp_1 = nn.Linear(embed_dim, hidden_dim)
        self.depth_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.gelu = nn.GELU()
        self.mlp_2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.mlp_1(x)
        x = x.permute(0,3,1,2) # B C H W
        x = self.depth_conv(x)
        x = x.permute(0,2,3,1) # B H W C
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.mlp_2(x)
        x = self.dropout(x)
        return x


class FFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.):
        super(FFN, self).__init__()
        self.mlp_1 = nn.Linear(embed_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.mlp_2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.mlp_1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.mlp_2(x)
        x = self.dropout(x)
        return x
    

class PPM_conv(nn.Module):
    def __init__(self, in_dim, ou_dim, pool_scales=(1,2,3,6)):
        super(PPM_conv,self).__init__()
        self.ppm = []
        for i in range(len(pool_scales)):
            self.ppm.append(nn.Sequential(
                                          nn.Conv2d(in_dim, 512, kernel_size=1, bias=False),
                                          nn.BatchNorm2d(512),
                                          nn.ReLU(inplace=True)
                                         ))
        self.ppm = nn.ModuleList(self.ppm)
        self.conv_last_ = nn.Sequential(
            nn.Conv2d(in_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, ou_dim, kernel_size=1)
            )

    def forward(self,x,xs):
        input_size = x.size()
        ppm_out = [x]
        for pool_scale,x_ in zip(self.ppm,xs):
            ppm_out.append(nn.functional.interpolate(
                pool_scale(x_),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last_(ppm_out)
        return x


class Clip_PSP_Block(nn.Module):
    def __init__(self, in_dim, out_dim, psp_weight=True, pool_scales=(1, 2, 3, 6)):
        super(Clip_PSP_Block, self).__init__()
        self.psp_weight = psp_weight
        self.pool_scales = pool_scales
        self.ppm_conv = PPM_conv(in_dim, out_dim, pool_scales=pool_scales)
        self.ppm_pool = []
        if self.psp_weight:
            self.pspweight_conv = nn.Sequential(nn.Conv2d(in_dim,1,kernel_size=1,bias=False),
                                                 nn.AdaptiveAvgPool2d((1,1)))
        for scale in pool_scales:
            self.ppm_pool.append(nn.AdaptiveAvgPool2d(scale))
        self.ppm_pool = nn.ModuleList(self.ppm_pool)

    def forward(self, feat, adj_feat):
        clip_num = len(adj_feat)
        adj_feat.append(feat)
        out_tmp = torch.cat(adj_feat, dim=0)

        if self.psp_weight:
            psp_w = self.pspweight_conv(out_tmp)
            psp_w = torch.split(psp_w, split_size_or_sections=int(psp_w.size(0)/(clip_num+1)), dim=0)
            psp_w = [psp_ww.unsqueeze(-1) for psp_ww in psp_w]
            psp_w = torch.cat(psp_w,dim=-1)
            psp_w = F.softmax(psp_w,dim=-1)

        out_tmp = torch.split(out_tmp, split_size_or_sections=int(out_tmp.size(0)/(clip_num+1)), dim=0)
        c_tmp = out_tmp[-1]
        others_tmp = out_tmp[:-1]
        pooled_features=[]
        for i in range(len(self.pool_scales)):
            pooled_features.append([])
        for i,pool in enumerate(self.ppm_pool):
            tmp_f = pool(c_tmp)
            pooled_features[i].append(tmp_f.unsqueeze(-1))
        for i,pool in enumerate(self.ppm_pool):
            for j,other in enumerate(others_tmp):
                tmp_f = pool(other)
                pooled_features[i].append(tmp_f.unsqueeze(-1))

        p_fs=[]
        for feature in pooled_features:
            feature = torch.cat(feature,dim=-1)
            if self.psp_weight:
#                psp_w = psp_w.expand_as(feature)
                feature = feature * psp_w
            feature = torch.mean(feature,dim=-1)
            p_fs.append(feature)
        pred_ = self.ppm_conv(c_tmp, p_fs)

        return pred_


import os
import pdb
import math
import torch
from torch import nn
from torch.autograd import Variable
def label_to_onehot(gt, num_classes, ignore_index=255):
    '''
    gt: ground truth with size (N, H, W)
    num_classes: the number of classes of different label
    '''
    N, H, W = gt.size()
    x = gt
    x[x == ignore_index] = num_classes
    # convert label into onehot format
    onehot = torch.zeros(N, x.size(1), x.size(2), num_classes + 1).cuda()
    onehot = onehot.scatter_(-1, x.unsqueeze(-1), 1)          

    return onehot.permute(0, 3, 1, 2)


class SpatialTemporalGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, scale=1):
        super(SpatialTemporalGather_Module, self).__init__()
        self.scale = scale
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feats, probs, clip_num ,memory=None,memory_num=None):
        assert(probs.size(0)==feats.size(0))
        probs_s = torch.split(probs,split_size_or_sections=int(probs.size(0)/(clip_num+1)), dim=0)
        feats_s = torch.split(feats,split_size_or_sections=int(feats.size(0)/(clip_num+1)), dim=0)
        if memory is None:
            contexts=[]
            for probs,feats in zip(probs_s,feats_s):
                batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
                probs = probs.view(batch_size, c, -1)
                feats = feats.view(batch_size, feats.size(1), -1)
                feats = feats.permute(0, 2, 1) # batch x hw x c 
                probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
                ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)# batch x k x c
                contexts.append(ocr_context.unsqueeze(0))
            contexts = torch.cat(contexts,dim=0)
            #contexts,_ = torch.max(contexts,dim=0)
            contexts = torch.mean(contexts,dim=0)
        else:
            if len(memory)>0:
                memory= [m.detach() for m in memory]
            for probs,feats in zip(probs_s,feats_s):
                batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
                probs = probs.view(batch_size, c, -1)
                feats = feats.view(batch_size, feats.size(1), -1)
                feats = feats.permute(0, 2, 1) # batch x hw x c 
                probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
                ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)# batch x k x c
                while len(memory)>memory_num:
                    memory.pop(0)
                memory.append(ocr_context.unsqueeze(0))
            contexts = torch.cat(memory,dim=0)
#            contexts,_ = torch.max(contexts,dim=0)
            contexts = torch.mean(contexts,dim=0)
#        print(len(memory))
            
          
        return contexts


class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        use_gt            : whether use the ground truth label map to compute the similarity map
        fetch_attention   : whether return the estimated similarity map
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 scale=1, 
                 use_gt=False,
                 use_bg=False,
                 fetch_attention=False
                 ):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.use_gt = use_gt
        self.use_bg = use_bg
        self.fetch_attention = fetch_attention
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True),
   
            #ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True),
#            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
#            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
#            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
#            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0),
#            ModuleHelper.BNReLU(self.in_channels, bn_type=bn_type),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, proxy, gt_label=None):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        if self.use_gt and gt_label is not None:
            gt_label = label_to_onehot(gt_label.squeeze(1).type(torch.cuda.LongTensor), proxy.size(2)-1)
            sim_map = gt_label[:, :, :, :].permute(0, 2, 3, 1).view(batch_size, h*w, -1)
            if self.use_bg:
                bg_sim_map = 1.0 - sim_map
                bg_sim_map = F.normalize(bg_sim_map, p=1, dim=-1)
            sim_map = F.normalize(sim_map, p=1, dim=-1)
        else:
            sim_map = torch.matmul(query, key)
            sim_map = (self.key_channels**-.5) * sim_map
            sim_map = F.softmax(sim_map, dim=-1)   

        # add bg context ...
        context = torch.matmul(sim_map, value) # hw x k x k x c
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=False)

        if self.use_bg:
            bg_context = torch.matmul(bg_sim_map, value)
            bg_context = bg_context.permute(0, 2, 1).contiguous()
            bg_context = bg_context.view(batch_size, self.key_channels, *x.size()[2:])
            bg_context = self.f_up(bg_context)
            bg_context = F.interpolate(input=bg_context, size=(h, w), mode='bilinear', align_corners=False)
            return context, bg_context
        else:
            if self.fetch_attention:
                return context, sim_map
            else:
                return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 scale=1, 
                 use_gt=False, 
                 use_bg=False,
                 fetch_attention=False
                 ):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale, 
                                                     use_gt,
                                                     use_bg,
                                                     fetch_attention
                                                     )


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.

    use_gt=True: whether use the ground-truth label to compute the ideal object contextual representations.
    use_bg=True: use the ground-truth label to compute the ideal background context to augment the representations.
    use_oc=True: use object context or not.
    """
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 out_channels, 
                 scale=1, 
                 dropout=0.1, 
                 use_gt=False,
                 use_bg=False,
                 use_oc=True,
                 fetch_attention=False
                 ):
        super(SpatialOCR_Module, self).__init__()
        self.use_gt = use_gt
        self.use_bg = use_bg
        self.use_oc = use_oc
        self.fetch_attention = fetch_attention
        self.object_context_block = ObjectAttentionBlock2D(in_channels, 
                                                           key_channels, 
                                                           scale, 
                                                           use_gt,
                                                           use_bg,
                                                           fetch_attention
                                                           )
        if self.use_bg:
            if self.use_oc:
                _in_channels = 3 * in_channels
            else:
                _in_channels = 2 * in_channels
        else:
            _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0),
            #ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats, gt_label=None):
        if self.use_gt and gt_label is not None:
            if self.use_bg:
                context, bg_context = self.object_context_block(feats, proxy_feats, gt_label)
            else:
                context = self.object_context_block(feats, proxy_feats, gt_label)
        else:
            if self.fetch_attention:
                context, sim_map = self.object_context_block(feats, proxy_feats)
            else:
                context = self.object_context_block(feats, proxy_feats)

        if self.use_bg:
            if self.use_oc:
                output = self.conv_bn_dropout(torch.cat([context, bg_context, feats], 1))
            else:
                output = self.conv_bn_dropout(torch.cat([bg_context, feats], 1))
        else:
            output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        if self.fetch_attention:
            return output, sim_map
        else:
            return output



class Clip_OCR_Block(nn.Module):
    """
    Object-Contextual Representations for Semantic Segmentation,
    Yuan, Yuhui and Chen, Xilin and Wang, Jingdong
    """
    def __init__(self, in_dim, out_dim, n_classes, use_memory=True):

        super(Clip_OCR_Block, self).__init__()
        self.use_memory = use_memory
        if use_memory:
            self.memory=[]
            self.memory_num = 5
        self.inplanes = 128
        self.n_classes = n_classes
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True)
        )

        self.spatial_context_head = SpatialTemporalGather_Module()
        self.spatial_ocr_head = SpatialOCR_Module(in_channels=in_dim, 
                                                  key_channels=in_dim//2, 
                                                  out_channels=in_dim,
                                                  scale=1,
                                                  dropout=0.05
                                                  )

        self.head = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn_head = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.05),
            nn.Conv2d(in_dim, n_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
        
    def forward(self, feat, adj_feat):
        clip_num = len(adj_feat)
        adj_feat.append(feat)
        x_dsn = self.dsn_head(torch.cat(adj_feat, dim=0))
        out_tmp = torch.cat(adj_feat, dim=0)
        out_tmp = self.conv_3x3(out_tmp)

        if self.use_memory:
            context = self.spatial_context_head(out_tmp, x_dsn, clip_num, self.memory, self.memory_num)
        else:
            context = self.spatial_context_head(out_tmp, x_dsn, clip_num)

        xs = torch.split(out_tmp,split_size_or_sections=int(out_tmp.size(0)/(clip_num+1)), dim=0)       
        x = xs[-1]
        x = self.spatial_ocr_head(x, context)
        x = self.head(x)
        return x

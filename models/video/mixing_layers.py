import torch
import torch.nn as nn
from einops import rearrange
from models.video.modules import (ShiftedWindows_MSA_3D, 
                                  ShiftedWindows_MSA_2D,
                                  Mix_FFN3D, 
                                  Mix_FFN2D)

# No mixing
class IdMixingBlock(nn.Module):
    def __init__(self):
        super(IdMixingBlock, self,).__init__()

    def forward(self, last_out_warped, out):
        return last_out_warped, out


# 3D-WMSA mixing
class STT3DMixingLayer(nn.Module):
    def __init__(self, embed_dim, window_size, shift_size, n_heads, hidden_dim, attn_drop=0., proj_drop=0., ffn_drop=0.):
        super(STT3DMixingLayer, self,).__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attention = ShiftedWindows_MSA_3D(embed_dim, window_size, shift_size, n_heads, attn_drop=attn_drop, proj_drop=proj_drop)
        self.ffn = Mix_FFN3D(embed_dim, hidden_dim=hidden_dim, dropout=ffn_drop)

    def forward(self, x):
        skip_1 = x
        x = self.norm(x)
        x = self.attention(x)
        x = x + skip_1

        skip_2 = x
        x = self.norm(x)
        x = self.ffn(x)
        x = x + skip_2
        return x
    
class STT3DMixingBlock(nn.Module):
    def __init__(self, embed_dim, n_layers, window_size, shift_size, n_heads, hidden_dim, attn_drop=0., proj_drop=0., ffn_drop=0.):
        super(STT3DMixingBlock, self,).__init__()
        self.embed_dim = embed_dim
        self.layers = nn.ModuleList([
            STT3DMixingLayer(embed_dim, window_size, shift_size, n_heads, hidden_dim, attn_drop, proj_drop, ffn_drop) for _ in range(n_layers)
        ])

    def forward(self, last_out_warped, out):
        x = torch.stack([last_out_warped, out], dim=1)
        x = rearrange(x, 'b t c h w -> b t h w c')
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        last_out_warped, out = rearrange(x[:,0], 'b h w c -> b c h w'), rearrange(x[:,1], 'b h w c -> b c h w')
        return last_out_warped, out
    

# 2D-WMSA mixing
class STT2DMixingLayer(nn.Module):
    def __init__(self, embed_dim, window_size, shift_size, n_heads, hidden_dim, attn_drop=0., proj_drop=0., ffn_drop=0.):
        super(STT2DMixingLayer, self,).__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attention = ShiftedWindows_MSA_2D(embed_dim, window_size, shift_size, n_heads, attn_drop=attn_drop, proj_drop=proj_drop)
        self.ffn = Mix_FFN2D(embed_dim, hidden_dim=hidden_dim, dropout=ffn_drop)

    def forward(self, x):
        skip_1 = x
        x = self.norm(x)
        x = self.attention(x)
        x = x + skip_1

        skip_2 = x
        x = self.norm(x)
        x = self.ffn(x)
        x = x + skip_2
        return x

class STT2DMixingBlock(nn.Module):
    def __init__(self, embed_dim, n_layers, window_size, shift_size, n_heads, hidden_dim, attn_drop=0., proj_drop=0., ffn_drop=0.):
        super(STT2DMixingBlock, self,).__init__()
        self.embed_dim = embed_dim
        self.layers = nn.ModuleList([
            STT2DMixingLayer(embed_dim, window_size, shift_size, n_heads, hidden_dim, attn_drop, proj_drop, ffn_drop) for _ in range(n_layers)
        ])

    def forward(self, last_out_warped, out):
        x = torch.cat([last_out_warped, out], dim=1)
        x = rearrange(x, 'b c h w -> b h w c')
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        last_out_warped, out = rearrange(x, 'b h w c -> b c h w')[:,:self.embed_dim//2], rearrange(x, 'b h w c -> b c h w')[:,self.embed_dim//2:]
        return last_out_warped, out
    

# 2D Conv mixing (inspired by convnext)
class Conv2DMixingLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(Conv2DMixingLayer, self,).__init__()
        self.dwconv = nn.Conv2d(embed_dim, embed_dim, kernel_size=7, padding=3, groups=embed_dim)  # depthwise conv
        self.layernorm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.pwconv1 = nn.Linear(embed_dim, hidden_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        skip = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.layernorm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        x = x + skip
        return x

class Conv2DMixingBlock(nn.Module):
    def __init__(self, embed_dim, n_layers, hidden_dim):
        super(Conv2DMixingBlock, self,).__init__()
        self.embed_dim = embed_dim
        self.layers = nn.ModuleList([
            Conv2DMixingLayer(embed_dim, hidden_dim) for _ in range(n_layers)
        ])

    def forward(self, last_out_warped, out):
        x = torch.cat([last_out_warped, out], dim=1)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        last_out_warped, out = x[:,:self.embed_dim//2], x[:,self.embed_dim//2:]
        return last_out_warped, out
    

# 3D Conv mixing
class Conv3DMixingLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(Conv3DMixingLayer, self,).__init__()
        self.dwconv = nn.Conv3d(embed_dim, embed_dim, kernel_size=7, padding=3, groups=embed_dim)  # depthwise conv
        self.layernorm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.pwconv1 = nn.Linear(embed_dim, hidden_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        skip = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, T, H, W) -> (N, T, H, W, C)
        x = self.layernorm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = x + skip
        return x

class Conv3DMixingBlock(nn.Module):
    def __init__(self, embed_dim, n_layers, hidden_dim):
        super(Conv3DMixingBlock, self,).__init__()
        self.embed_dim = embed_dim
        self.layers = nn.ModuleList([
            Conv3DMixingLayer(embed_dim, hidden_dim) for _ in range(n_layers)
        ])

    def forward(self, last_out_warped, out):
        x = torch.stack([last_out_warped, out], dim=2)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        last_out_warped, out = x[:,:,0], x[:,:,1]
        return last_out_warped, out

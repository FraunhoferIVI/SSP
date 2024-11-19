import torch
import torch.nn as nn
from einops import rearrange

from models.video.mixing_layers import (
    STT3DMixingLayer,
    STT2DMixingLayer,
    Conv2DMixingLayer,
)

# Constant interpolation. If alpha=1, no interpolation, only current prediction.
class ConstantSimBlock(nn.Module):
    def __init__(self, alpha):
        super(ConstantSimBlock, self).__init__()
        self.alpha = alpha

    def forward(self, last_featmap, featmap, size):
        sim = torch.ones((featmap.size(0), 1, size[0], size[1]), dtype=torch.float32, device=featmap.device) * self.alpha
        return sim

# Constant interpolation with learnable value. Starts at 0.5 for average of both predictions
class AverageSimBlock(nn.Module):
    def __init__(self, alpha):
        super(AverageSimBlock, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, last_featmap, featmap, size):
        sim = torch.ones((featmap.size(0), 1, size[0], size[1]), dtype=torch.float32, device=featmap.device) * self.alpha
        return sim

# Cosine similarity on features
class CossimSimBlock(nn.Module):
    def __init__(self):
        super(CossimSimBlock, self,).__init__()
        self.cossim = nn.CosineSimilarity(dim=1)

    def forward(self, last_featmap, featmap, size):
        sim = self.cossim(featmap, last_featmap)
        sim = sim.unsqueeze(1)
        sim = torch.clamp(sim, min=0, max=1)**4
        sim = torch.clamp(sim, min=0, max=0.95)
        if sim.shape[-2:] != size:
            sim = nn.functional.interpolate(sim, size=size, mode="bilinear", align_corners=False)
        return sim

# Convolutions on last features
class ConvSimBlockBase(nn.Module):
    def __init__(self, embed_dim):
        super(ConvSimBlockBase, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim//2, 3, padding=1),
            nn.BatchNorm2d(embed_dim//2),
            nn.ReLU(),
            nn.Conv2d(embed_dim//2,1,1)
        )

    def forward(self, last_featmap, featmap, size):
        x = last_featmap
        x = self.conv(x)
        sim = x.sigmoid()
        if sim.shape[-2:] != size:
            sim = nn.functional.interpolate(sim, size=size, mode="bilinear", align_corners=False)
        return sim

# Convolutions on concatenated features
class ConvSimBlock(nn.Module):
    def __init__(self, embed_dim):
        super(ConvSimBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim*2, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim//2, 3, padding=1),
            nn.BatchNorm2d(embed_dim//2),
            nn.ReLU(),
            nn.Conv2d(embed_dim//2,1,1)
        )

    def forward(self, last_featmap, featmap, size):
        x = torch.cat([last_featmap, featmap], dim=1)
        x = self.conv(x)
        sim = x.sigmoid()
        if sim.shape[-2:] != size:
            sim = nn.functional.interpolate(sim, size=size, mode="bilinear", align_corners=False)
        return sim

# Mixing of features with cnvolution then convolution on last features
class ConvSimBlockv2(nn.Module):
    def __init__(self, embed_dim):
        super(ConvSimBlockv2, self).__init__()
        self.convmix = Conv2DMixingLayer(embed_dim*2, hidden_dim=8*embed_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim//2, 3, padding=1),
            nn.BatchNorm2d(embed_dim//2),
            nn.ReLU(),
            nn.Conv2d(embed_dim//2,1,1)
        )
        self.embed_dim = embed_dim

    def forward(self, last_featmap, featmap, size):
        x = torch.cat([last_featmap, featmap], dim=1)
        x = self.convmix(x)
        x = x[:,:self.embed_dim]
        x = self.conv(x)
        sim = x.sigmoid()
        if sim.shape[-2:] != size:
            sim = nn.functional.interpolate(sim, size=size, mode="bilinear", align_corners=False)
        return sim
    

# 3D-WMSA mixing then convolution on last features
class STT3DConvSimBlock(nn.Module):
    def __init__(self, embed_dim):
        super(STT3DConvSimBlock, self).__init__()
        self.att = STT3DMixingLayer(embed_dim, window_size=[2,7,7], shift_size=[0,0,0], n_heads=8, hidden_dim=embed_dim*4)
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim//2, 3, padding=1),
            nn.BatchNorm2d(embed_dim//2),
            nn.ReLU(),
            nn.Conv2d(embed_dim//2,1,1)
        )

    def forward(self, last_featmap, featmap, size):
        x = torch.stack([last_featmap, featmap], dim=1)
        x = rearrange(x, 'b t c h w -> b t h w c')
        x = self.att(x)
        x = self.conv(rearrange(x[:,0], 'b h w c -> b c h w'))
        sim = x.sigmoid()
        if sim.shape[-2:] != size:
            sim = nn.functional.interpolate(sim, size=size, mode="bilinear", align_corners=False)
        return sim
    
# 3D-WMSA mixing then convolution on concatenated features
class STT3DConvSimBlock2(nn.Module):
    def __init__(self, embed_dim):
        super(STT3DConvSimBlock2, self).__init__()
        self.att = STT3DMixingLayer(embed_dim, window_size=[2,7,7], shift_size=[0,0,0], n_heads=8, hidden_dim=embed_dim*4)
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim*2, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
            nn.Conv2d(embed_dim,1,1)
        )

    def forward(self, last_featmap, featmap, size):
        x = torch.stack([last_featmap, featmap], dim=1)
        x = rearrange(x, 'b t c h w -> b t h w c')
        x = self.att(x)
        x = self.conv(torch.cat([rearrange(x[:,0], 'b h w c -> b c h w'), rearrange(x[:,1], 'b h w c -> b c h w')], dim=1))
        sim = x.sigmoid()
        if sim.shape[-2:] != size:
            sim = nn.functional.interpolate(sim, size=size, mode="bilinear", align_corners=False)
        return sim
    
# 3D-WMSA mixing then cosine similarity
class STT3DCosSimBlock(nn.Module):
    def __init__(self, embed_dim):
        super(STT3DCosSimBlock, self).__init__()
        self.att = STT3DMixingLayer(embed_dim, window_size=[2,7,7], shift_size=[0,0,0], n_heads=8, hidden_dim=embed_dim*4)
        self.cossim = nn.CosineSimilarity(dim=1)

    def forward(self, last_featmap, featmap, size):
        x = torch.stack([last_featmap, featmap], dim=1)
        x = rearrange(x, 'b t c h w -> b t h w c')
        sim = self.cossim(rearrange(x[:,0], 'b h w c -> b c h w'), rearrange(x[:,1], 'b h w c -> b c h w'))
        sim = sim.unsqueeze(1)
        sim = torch.clamp(sim, min=0, max=1)**4
        sim = torch.clamp(sim, min=0, max=0.99)
        if sim.shape[-2:] != size:
            sim = nn.functional.interpolate(sim, size=size, mode="bilinear", align_corners=False)
        return sim
import torch
import torch.nn as nn
from RAFT_core.raft import RAFT
from collections import OrderedDict

def flowwarp(x, flo, mode="nearest"):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.to(x.device)
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)
    output = nn.functional.grid_sample(x, vgrid,mode=mode,align_corners=False)

    return output

def get_flow_model():
    model_raft = RAFT()
    to_load = torch.load('./RAFT_core/raft-things.pth-no-zip', weights_only=True)
    new_state_dict = OrderedDict()
    for k, v in to_load.items():
        name = k[7:] 
        new_state_dict[name] = v 
    model_raft.load_state_dict(new_state_dict)
    return model_raft


def interpolate_flow(flow, factor):
    return nn.functional.interpolate(flow, scale_factor=factor, mode="bilinear", align_corners=False)*factor
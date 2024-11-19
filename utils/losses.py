import torch
import torch.nn as nn

from torch import Tensor
from models.opt_flow import flowwarp


class KLDivSoftLabelLoss(nn.Module):
    def __init__(self, reduction: str = 'batchmean', softmax_on_target: bool = False, temperature: int = 1):
        if reduction not in ('mean', 'sum', 'none', 'batchmean'):
            raise ValueError(
                'Reduction must be one of: "batchmean", "mean", "sum", "none".')

        super().__init__()
        self.reduction = reduction
        self.kldivloss = nn.KLDivLoss(reduction=reduction)
        self.softmax_on_target = softmax_on_target
        self.temperature = temperature

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = nn.functional.log_softmax(x / self.temperature, dim=1)
        if self.softmax_on_target:
            y = nn.functional.softmax(y / self.temperature, dim=1)
        return self.kldivloss(x, y) * (self.temperature ** 2)


class TempConstLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights
    def forward(self, out, last_out, flow_model, frame, last_frame):
        with torch.no_grad():
            flow_model.eval()
            _, flow = flow_model(frame, last_frame, iters=20, test_mode=True)
            flow = flow.detach()
        occlusion_mask = torch.exp(-torch.abs((frame - flowwarp(last_frame, flow)).sum(1, keepdim=True)))
        warped_last_out = flowwarp(last_out, flow)
        if self.weights is not None:
            temp_const_loss = ((occlusion_mask * (out - warped_last_out)**2)*self.weights).mean()
        else:
            temp_const_loss = (occlusion_mask * (out - warped_last_out)**2).mean()
        temp_const_loss = (occlusion_mask * (out - warped_last_out)**2).mean()
        return temp_const_loss

            

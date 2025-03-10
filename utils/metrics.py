from torchmetrics import JaccardIndex
import numpy as np
import torch
import torch.nn as nn
from functools import wraps
from time import perf_counter


class Timing:
    """Class to time functions and methods."""

    def __init__(self):
        self.inf_time = []

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = perf_counter()
            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)
            # start.record()
            result = func(*args, **kwargs)
            # end.record()
            # torch.cuda.synchronize()
            end = perf_counter()
            self.inf_time.append(end - start)
            # self.inf_time.append(start.elapsed_time(end) / 1000)  # in seconds
            return result

        return wrapper

    def print_average(self):
        """Print the average timing of the function."""
        print(f"---\tFunction took {np.mean(self.inf_time):.4f} seconds to run!")

    def get_average(self) -> float:
        """Return the average timing of the function."""
        return float(np.mean(self.inf_time))

    def reset(self):
        """Reset the timing of the function."""
        self.inf_time = []



class MetricMeter():
    def __init__(self):
        self.reset()
    def reset(self):
        self.count = 0
        self.last_value = 0
        self.sum = 0
        self.avg = 0
    def update(self, value):
        self.count += 1
        self.last_value = value
        self.sum += value
        self.avg = self.sum / self.count

class PerClassMetricMeter():
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.reset()
    def reset(self):
        self.counts = np.zeros(self.n_classes)
        self.last_values = np.zeros(self.n_classes)
        self.sum = np.zeros(self.n_classes)
        self.avg = np.zeros(self.n_classes)
    def update(self, values, valid_classes):
        self.counts += valid_classes
        self.last_values = values
        self.sum += values
        self.avg = (self.sum / (self.counts + 1e-6))
        self.avg[self.counts==0] = 0

def pixel_accuracy(preds, labels):
    acc = (preds == labels).sum()/(labels>=0).sum()
    return acc

def meanIoU(preds, labels, n_classes, ignore_index):
    mIoU = JaccardIndex(task="multiclass", num_classes=n_classes, average="macro", ignore_index=ignore_index)
    return mIoU(preds, labels)

def weightedIoU(preds, labels, n_classes, ignore_index):
    wIoU = JaccardIndex(task="multiclass", num_classes=n_classes, average="weighted", ignore_index=ignore_index)
    return wIoU(preds, labels)

def class_meanIoU(preds, labels, n_classes, ignore_index):
    mIoU = JaccardIndex(task="multiclass", num_classes=n_classes, average="none", ignore_index=ignore_index)
    return mIoU(preds, labels)


#https://github.com/VSPW-dataset/VSPW_code/blob/main/VC_perclip.py
def video_consistency(label_list, pred_list, n_frames):
    h, w = label_list[0].shape
    accs = []
    for i in range(len(label_list) - n_frames):
        global_common = np.ones((h,w))
        predglobal_common = np.ones((h,w))

        for j in range(1, n_frames):
            common = (label_list[i] == label_list[i+j])
            global_common = np.logical_and(global_common,common)
            pred_common = (pred_list[i] == pred_list[i+j])
            predglobal_common = np.logical_and(predglobal_common,pred_common)
        pred = (predglobal_common*global_common)

        acc = pred.sum()/global_common.sum()
        accs.append(acc)
    return accs

#https://github.com/princeton-vl/RAFT/tree/master
from RAFT_core.raft import RAFT
from collections import OrderedDict
def get_flow_model():
    model_raft = RAFT()
    to_load = torch.load('./RAFT_core/raft-things.pth-no-zip', weights_only=True)
    new_state_dict = OrderedDict()
    for k, v in to_load.items():
        name = k[7:] 
        new_state_dict[name] = v 
    model_raft.load_state_dict(new_state_dict)
    return model_raft

#https://github.com/VSPW-dataset/VSPW_code/blob/main/TC_cal.py
def flowwarp(x, flo):
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
    output = nn.functional.grid_sample(x, vgrid,mode='nearest',align_corners=False)

    return output

def temporal_consistency(frame_list, pred_list, model_raft, n_classes, device):
    tc = 0
    for (i, frame) in enumerate(frame_list[:-1]):
        frame = torch.from_numpy(frame).unsqueeze(0).to(device)
        next_frame = frame_list[i+1]
        next_frame = torch.from_numpy(next_frame).unsqueeze(0).to(device)

        with torch.no_grad():
            model_raft.eval()
            _, flow = model_raft(frame, next_frame, iters=20, test_mode=True)

        flow = flow.detach().cpu()

        pred = pred_list[i]
        next_pred = pred_list[i+1]
        pred = torch.from_numpy(pred)
        next_pred = torch.from_numpy(next_pred)
        next_pred = next_pred.unsqueeze(0).unsqueeze(0).float()
        warp_pred = flowwarp(next_pred,flow)
        warp_pred = warp_pred.int().squeeze(1)
        pred = pred.unsqueeze(0)

        tc += meanIoU(warp_pred, pred, n_classes=n_classes, ignore_index=255)

    tc = tc/len(frame_list[:-1])
    return tc

# new temporal consistency metrics based on: https://arxiv.org/pdf/2308.07615
def temporal_consistency_proposed(frame_list, pred_list, model_raft, n_classes, device):
    """
    Compute temporal consistency for segmentation.
    For each center frame (of a 5-frame chain), warp the segmentation predictions
    from the two frames before and after into the center frame’s coordinate system.
    Then, for each pixel, the consistency is the percentage of the 5 votes
    that agree with the majority class.
    
    Args:
      frame_list: list of frames (as numpy arrays).
      pred_list: list of segmentation predictions (numpy arrays of shape HxW with integer class labels).
      model_raft: optical flow model.
      n_classes: number of segmentation classes.
      device: torch device.
      
    Returns:
      Average segmentation consistency (a value between 0 and 1).
    """
    chain_length = 5
    consistency_total = 0.0
    num_chains = 0
    
    # We need at least 2 frames before and 2 after the center
    for i in range(2, len(frame_list) - 2):
        # Get height and width from the center prediction
        H, W = pred_list[i].shape
        # List to collect predictions for the 5-frame chain
        chain_preds = []
        
        # Build the temporal chain: frames i-2, i-1, i, i+1, i+2
        for offset in range(-2, 3):
            j = i + offset
            if j == i:
                # Center frame: no warping needed.
                pred_center = torch.from_numpy(pred_list[j])
                chain_preds.append(pred_center.unsqueeze(0))  # shape: (1, H, W)
            else:
                # Warp the prediction from frame j to frame i.
                src_frame = torch.from_numpy(frame_list[j]).unsqueeze(0).to(device)
                tgt_frame = torch.from_numpy(frame_list[i]).unsqueeze(0).to(device)
                with torch.no_grad():
                    model_raft.eval()
                    # Compute optical flow from source (j) to target (i)
                    _, flow = model_raft(tgt_frame, src_frame, iters=20, test_mode=True)
                flow = flow.detach().cpu()
                # Get the prediction for frame j and add channel dims: (1,1,H,W)
                pred_j = torch.from_numpy(pred_list[j]).unsqueeze(0).unsqueeze(0).float()
                warped_pred = flowwarp(pred_j, flow)  # assume shape remains (1,1,H,W)
                # Remove extra dims to get (H,W)
                warped_pred = warped_pred.squeeze(0).squeeze(0).long()
                chain_preds.append(warped_pred.unsqueeze(0))
        
        # Stack into tensor: shape (5, H, W)
        chain_tensor = torch.cat(chain_preds, dim=0)
        
        # Compute the consistency map per pixel.
        consistency_map = torch.zeros((H, W))
        # Loop over each pixel (this can be vectorized for efficiency)
        chain_np = chain_tensor.cpu().numpy()  # shape (5, H, W)
        for y in range(H):
            for x in range(W):
                pixel_chain = chain_np[:, y, x]
                # Count the frequency of each label along the chain.
                values, counts = np.unique(pixel_chain, return_counts=True)
                max_votes = counts.max()
                consistency_map[y, x] = max_votes / chain_length
        
        consistency_total += consistency_map.mean().item()
        num_chains += 1
        print(f"Chain {num_chains}: consistency = {consistency_map.mean().item():.4f}")
        print(consistency_total)
    
    return consistency_total / num_chains if num_chains > 0 else 0


def temporal_consistency_forwardbackward(frame_list, pred_list, model_raft, n_classes, device):
    tc = 0
    for (i, frame) in enumerate(frame_list[:-1]):
        frame = torch.from_numpy(frame).unsqueeze(0).to(device)
        next_frame = frame_list[i+1]
        next_frame = torch.from_numpy(next_frame).unsqueeze(0).to(device)

        with torch.no_grad():
            model_raft.eval()
            _, flow_backward = model_raft(frame, next_frame, iters=15, test_mode=True)
            _, flow_forward = model_raft(next_frame, frame, iters=15, test_mode=True)

        flow_backward = flow_backward.detach().cpu()
        flow_forward = flow_forward.detach().cpu()

        occlusion_mask = ((flow_backward + flow_forward)**2).sum(1) > 0.01*((flow_backward**2).sum(1) + (flow_forward**2).sum(1)) + 0.5

        pred = pred_list[i]
        next_pred = pred_list[i+1]
        pred = torch.from_numpy(pred)
        next_pred = torch.from_numpy(next_pred)
        pred = pred.unsqueeze(0).unsqueeze(0).float()
        next_pred = next_pred.unsqueeze(0).unsqueeze(0).float()

        warp_next_pred = flowwarp(next_pred, flow_backward)
        warp_pred = flowwarp(pred, flow_forward)

        warp_next_pred = warp_next_pred.int().squeeze(1)
        warp_pred = warp_pred.int().squeeze(1)
        pred = pred.int().squeeze(1)
        next_pred = next_pred.int().squeeze(1)

        tc += (((~occlusion_mask)*(warp_next_pred==pred)).sum()/(~occlusion_mask).sum() + ((~occlusion_mask)*(warp_pred==next_pred)).sum()/(~occlusion_mask).sum())/2

    tc = tc/len(frame_list[:-1])
    return tc

def temporal_consistency_bidirectionnal(frame_list, pred_list, model_raft, n_classes, device):
    tc = 0
    for (i, frame) in enumerate(frame_list[1:-1]):
        frame = torch.from_numpy(frame).unsqueeze(0).to(device)
        next_frame = frame_list[i+1]
        next_frame = torch.from_numpy(next_frame).unsqueeze(0).to(device)
        previous_frame = frame_list[i-1]
        previous_frame = torch.from_numpy(previous_frame).unsqueeze(0).to(device)

        with torch.no_grad():
            model_raft.eval()
            _, flow_next = model_raft(frame, next_frame, iters=15, test_mode=True)
            _, flow_previous = model_raft(frame, previous_frame, iters=15, test_mode=True)

        flow_next = flow_next.detach().cpu()
        flow_previous = flow_previous.detach().cpu()

        pred = pred_list[i]
        next_pred = pred_list[i+1]
        previous_pred = pred_list[i-1]
        pred = torch.from_numpy(pred)
        next_pred = torch.from_numpy(next_pred)
        previous_pred = torch.from_numpy(previous_pred)
        next_pred = next_pred.unsqueeze(0).unsqueeze(0).float()
        previous_pred = previous_pred.unsqueeze(0).unsqueeze(0).float()

        warp_next_pred = flowwarp(next_pred, flow_next)
        warp_previous_pred = flowwarp(previous_pred, flow_previous)

        warp_next_pred = warp_next_pred.int().squeeze(1)
        warp_previous_pred = warp_previous_pred.int().squeeze(1)
        pred = pred.unsqueeze(0)

        tc += ((warp_next_pred==pred).sum() + (warp_previous_pred==pred).sum())/2

    tc = tc/len(frame_list[1:-1])
    return tc


def temporal_consistency_nframes(frame_list, pred_list, model_raft, n_classes, device, n_window):
    tc = 0
    for (i, frame) in enumerate(frame_list[:-1]):
        frame = torch.from_numpy(frame).unsqueeze(0).to(device)
        adj_frames = frame_list[max(i-n_window, 0):i] + frame_list[i+1:min(i+n_window+1, len(frame_list))]
        for next_frame in adj_frames:
            next_frame = torch.from_numpy(next_frame).unsqueeze(0).to(device)

            with torch.no_grad():
                model_raft.eval()
                _, flow = model_raft(frame, next_frame, iters=20, test_mode=True)

            flow = flow.detach().cpu()

            pred = pred_list[i]
            next_pred = pred_list[i+1]
            pred = torch.from_numpy(pred)
            next_pred = torch.from_numpy(next_pred)
            next_pred = next_pred.unsqueeze(0).unsqueeze(0).float()
            warp_pred = flowwarp(next_pred,flow)
            warp_pred = warp_pred.int().squeeze(1)
            pred = pred.unsqueeze(0)

            tc += meanIoU(warp_pred, pred, n_classes=n_classes, ignore_index=255)/len(adj_frames)

    tc = tc/len(frame_list[:-1])
    return tc


def temporal_consistency_kskip(frame_list, pred_list, model_raft, n_classes, device, k_skip):
    tc = 0
    for (i, frame) in enumerate(frame_list[:-k_skip]):
        frame = torch.from_numpy(frame).unsqueeze(0).to(device)
        next_frame = frame_list[i+k_skip]
        next_frame = torch.from_numpy(next_frame).unsqueeze(0).to(device)

        with torch.no_grad():
            model_raft.eval()
            _, flow = model_raft(frame, next_frame, iters=20, test_mode=True)

        flow = flow.detach().cpu()

        pred = pred_list[i]
        next_pred = pred_list[i+1]
        pred = torch.from_numpy(pred)
        next_pred = torch.from_numpy(next_pred)
        next_pred = next_pred.unsqueeze(0).unsqueeze(0).float()
        warp_pred = flowwarp(next_pred,flow)
        warp_pred = warp_pred.int().squeeze(1)
        pred = pred.unsqueeze(0)

        tc += meanIoU(warp_pred, pred, n_classes=n_classes, ignore_index=255)

    tc = tc/len(frame_list[:-1])
    return tc

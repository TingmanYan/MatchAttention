from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.convformer import LayerNormWithoutBias
from utils.utils import init_coords

class GlobalCorrelation(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNormWithoutBias(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.scale = dim**-0.5

    def forward(self, x, stereo=True):
        x = self.norm(x)
        ref, tgt = x.chunk(2, dim=0)
        ref, tgt = self.q(ref), self.k(tgt)
        # global correlation on horizontal direction
        B, H, W, C = ref.shape

        if stereo:
            correlation = torch.matmul(ref, tgt.transpose(-2, -1))*self.scale  # [B, H, W, W]

            # mask subsequent positions to make disparity positive
            mask = torch.triu(torch.ones((W, W), dtype=ref.dtype, device=ref.device), diagonal=1) # [W, W]
            valid_mask = (mask == 0).unsqueeze(0).unsqueeze(0).repeat(B, H, 1, 1)  # [B, H, W, W]

            mask_ = torch.triu(torch.ones((W, W), dtype=ref.dtype, device=ref.device), diagonal=0) # mask for input order [right, left]
            valid_mask_ = (mask_ != 0).unsqueeze(0).unsqueeze(0).repeat(B, H, 1, 1) # upper right
            valid_mask = torch.cat((valid_mask, valid_mask_), dim=0) # [B*2, H, W, W]
            correlation = torch.cat((correlation, correlation.permute(0, 1, 3, 2)), dim=0) # [B*2, H, W, W]
            B = B*2

            correlation[~valid_mask] = -1e9 if correlation.dtype == torch.float32 else -1e4

            # build volume from correlation
            D = W # all-pair correlation
            volume = correlation.new_zeros([B, D, H, W])
            for d in range(D): # most time-consuming
                volume[:B//2, d, :, d:] = correlation[:B//2, :, range(d, W), range(W-d)]
                volume[B//2:, d, :, :(W-d)] = correlation[B//2:, :, range(W-d), range(d, W)]

            volume = F.softmax(volume, dim=1).to(volume.dtype)

            volume_clone = volume.clone()
            for d in range(D): # fill out of view # second time-consuming
                volume_clone[:B//2, d, :, :d] = volume[:B//2, d, :, d:d+1] # left
                volume_clone[B//2:, d, :, W-1-d:] = volume[B//2:, d, :, W-1-d:(W-d)] # right

            flow = local_disparity_estimator(volume_clone)
            return flow, volume_clone
        else:
            init_grid = init_coords(ref) # [B, H, W, 2]
            ref = ref.view(B, -1, C)  # [B, H*W, C]
            tgt = tgt.view(B, -1, C)  # [B, H*W, C]

            correlation = torch.matmul(ref, tgt.transpose(-2, -1))*self.scale  # [B, H*W, H*W]
            correlation = torch.cat((correlation, correlation.permute(0, 2, 1)), dim=0)  # [2*B, H*W, H*W]
            init_grid = init_grid.repeat(2, 1, 1, 1)  # [2*B, H, W, 2]
            B = B * 2

            prob = F.softmax(correlation, dim=-1).to(correlation.dtype)  # [B, H*W, H*W]

            flow = local_flow_estimator(prob, init_grid)

            return flow, prob.view(B, H, W, H*W)

def local_flow_estimator(prob, init_grid, k=5):
    """
    Flow estimator using weighted sum within local window centered at max prob
    Args:
        prob: normalized correlation volume [B, H*W, H*W]
        init_grid: init coordinate grid [B, H, W, 2]
        k: local window size (odd number)
    Returns:
        flow: optical field [B, H, W, 2]
    """
    B, H, W, _ = init_grid.shape
    r = k // 2
    device = prob.device

    prob_blur = F.avg_pool2d(prob, kernel_size=k, stride=1, padding=r).view(B, H*W, H*W)
    
    max_prob, max_idx = torch.max(prob_blur, dim=-1)  # [B, H*W]
    max_idx = max_idx.unsqueeze(-1)  # [B, H*W, 1]
    target_coords = init_grid  # [B, H, W, 2]
    max_y = max_idx // W  # [B, H*W, 1]
    max_x = max_idx % W   # [B, H*W, 1]
    max_y = torch.clamp(max_y, r, H-1-r) 
    max_x = torch.clamp(max_x, r, W-1-r)
    
    yy, xx = torch.meshgrid(torch.arange(-r, r+1, device=device), torch.arange(-r, r+1, device=device), indexing='ij')
    offsets_y = yy.reshape(1, 1, k*k, 1)  # [1, 1, k*k, 1]
    offsets_x = xx.reshape(1, 1, k*k, 1)  # [1, 1, k*k, 1]
    sample_y = max_y.unsqueeze(2) + offsets_y  # [B, H*W, k*k, 1]
    sample_x = max_x.unsqueeze(2) + offsets_x  # [B, H*W, k*k, 1]
    sample_y = sample_y.long().squeeze(-1)  # [B, H*W, k*k]
    sample_x = sample_x.long().squeeze(-1)  # [B, H*W, k*k]
    
    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(-1, H*W, k*k)
    window_coords = target_coords[batch_idx, sample_y, sample_x]  # [B, H*W, k*k, 2]
    
    window_indices = sample_y * W + sample_x  # [B, H*W, k*k]
    window_probs = torch.gather(prob, dim=-1, index=window_indices)  # [B, H*W, k*k]
    
    mean_prob = 1.0 / (H * W)
    invalid_mask = window_probs < mean_prob
    window_probs[invalid_mask] = 0
    
    window_probs_sum = window_probs.sum(dim=-1, keepdim=True).to(window_probs.dtype)
    window_probs_sum = torch.clamp(window_probs_sum, min=torch.finfo(window_probs_sum.dtype).tiny)
    normalized_probs = window_probs / window_probs_sum  # [B, H*W, k*k]
    normalized_probs = normalized_probs.unsqueeze(-1)  # [B, H*W, k*k, 1]
    correspondence = torch.sum(normalized_probs * window_coords, dim=2).to(normalized_probs.dtype) # [B, H*W, 2]
    correspondence = correspondence.view(B, H, W, 2)  # [B, H, W, 2]
    flow = correspondence - init_grid
    
    return flow

def local_disparity_estimator(cv, k=5):
    """
    Disparity estimator using weighted sum within local window centered at max prob
    Args:
        cv: cost volume [B, D, H, W]
        k: local window size (odd number)
    Returns:
        flow: [B, H, W, 2]
    """
    B, D, H, W = cv.shape
    r = k // 2
    device = cv.device
    
    cv_blur = F.avg_pool1d(cv.permute(0, 2, 3, 1).view(B, -1, D), kernel_size=k, stride=1, padding=r).view(B, H, W, D).permute(0, 3, 1, 2)
    
    # find max idx in blured cv
    max_cv, max_idx = torch.max(cv_blur, dim=1)  # max_idx: [B, H, W]
    max_idx = max_idx.unsqueeze(1)  # [B, 1, H, W]
    max_idx = torch.clamp(max_idx, r, D-1-r)  # [B, 1, H, W]
    
    offsets = torch.arange(-r, r+1, device=device).view(1, k, 1, 1)  # [1, k, 1, 1]
    
    sample_idx = max_idx + offsets  # [B, k, H, W]
    sample_idx = torch.clamp(sample_idx, 0, D-1)
    
    batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(-1, k, H, W)
    h_idx = torch.arange(H, device=device).view(1, 1, H, 1).expand(B, k, H, W)
    w_idx = torch.arange(W, device=device).view(1, 1, 1, W).expand(B, k, H, W)
    
    window_probs = cv[batch_idx, sample_idx, h_idx, w_idx]  # [B, k, H, W]
    
    mean_prob = 1.0 / D
    invalid_mask = window_probs < mean_prob
    window_probs[invalid_mask] = 0
    
    # normalize within local window
    window_probs_sum = window_probs.sum(dim=1, keepdim=True).to(window_probs.dtype)  # [B, 1, H, W]
    window_probs_sum = torch.clamp(window_probs_sum, min=torch.finfo(window_probs_sum.dtype).tiny)
    normalized_probs = window_probs / window_probs_sum  # [B, k, H, W]
    
    window_disp = sample_idx.to(normalized_probs.dtype) # [B, k, H, W]
    
    disp = torch.sum(normalized_probs * window_disp, dim=1).to(normalized_probs.dtype).unsqueeze(-1)  # [B, H, W, 1]
    
    return disp_to_flow(disp, B)

def disp_to_flow(disp, B):
    ## disp[:B//2, ...] = -disp[:B//2, ...] # negetive left flow

    ## for onnx support
    batch_indices = torch.arange(B, device=disp.device)
    mask = batch_indices < (B // 2)
    
    disp = torch.where(mask.view(B, 1, 1, 1), -disp, disp)

    flow = torch.cat((disp, torch.zeros_like(disp)), dim=-1).contiguous() # [B, H, W, 2]
    return flow

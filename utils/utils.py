import torch
import torch.nn.functional as F
import numpy as np


class InputPadder:
    """ Pads images such that dimensions are divisible by padding_factor """

    def __init__(self, dims, mode='top_right', padding_factor=32):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // padding_factor) + 1) * padding_factor - self.ht) % padding_factor
        pad_wd = (((self.wd // padding_factor) + 1) * padding_factor - self.wd) % padding_factor
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        elif mode == 'top_right':
            self._pad = [0, pad_wd, pad_ht, 0]
        elif mode == 'bottom_right':
            self._pad = [0, pad_wd, 0, pad_ht]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def init_coords(ref):
    B, H, W, C = ref.shape

    coords = torch.meshgrid(torch.arange(H, device=ref.device, dtype=ref.dtype), torch.arange(W, device=ref.device, dtype=ref.dtype), indexing='ij')
    coords = torch.stack(coords[::-1], dim=-1)
    return coords[None].repeat(B, 1, 1, 1).to(ref.device) # [B, H, W, 2]


def bilinear_sample_by_offset(tgt, offset): # tgt [B, _, H, W], offset [B, H, W, 2]
    _, _, H, W = tgt.shape

    xgrid, ygrid = offset.split([1, 1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)

    tgt_to_ref = F.grid_sample(tgt, grid, mode='bilinear', align_corners=True)
    return tgt_to_ref

def calc_noc_mask(field, A=2):
    offset = field + init_coords(field) # [B, H, W, 2]
    field_ref_, field_tgt_ = field.chunk(2, dim=0)
    field_ref = torch.cat((field_ref_, field_tgt_), dim=0) # order
    field_tgt = torch.cat((field_tgt_, field_ref_), dim=0) # reverse order
    field_tgt_to_ref = bilinear_sample_by_offset(field_tgt.permute(0, 3, 1, 2).contiguous(), offset).permute(0, 2, 3, 1).contiguous()
    field_diff = torch.abs(field_ref + field_tgt_to_ref).sum(dim=-1) # ref and tgt flow has different sign
    noc_mask = (field_diff < A).to(field_diff.dtype)
    return noc_mask
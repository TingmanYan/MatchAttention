import torch
import torch.nn as nn

from timm.models.layers import DropPath

from models.convformer import LayerNormWithoutBias
from models.common import ConvGLU
from models.mat_pytorch_impl import compute_bilinear_weights, compute_match_attention, compute_bilinear_softmax, attention_aggregate
from models.match_former_ops import MF_FusedForwardOps
from utils.utils import bilinear_sample_by_offset, init_coords

class MatchAttention(torch.nn.Module):
    r"""MatchAttention: Matching the relative positions
    """
    def __init__(self, args, dim, win_r=[1, 1], num_head=8, head_dim=None, qkv_bias=False, 
                 attn_drop=0., proj_drop=0., proj_bias=False, cross=False, noc_embed=False, **kargs):
        super().__init__()

        self.num_head = num_head
        self.cross = cross
        self.noc_embed = noc_embed if not cross else False # only for self attention

        self.head_dim = dim // num_head if head_dim is None else head_dim
        self.scale = self.head_dim ** -0.5

        self.attention_dim = self.num_head * self.head_dim

        self.win_r = win_r
        self.attn_num = (2*win_r[0]+2)*(2*win_r[1]+2)

        embed_dim = dim + 1 if noc_embed else dim # '1' for noc_mask
        self.q = nn.Linear(embed_dim, self.attention_dim, bias=qkv_bias)
        self.k = nn.Linear(embed_dim, self.attention_dim, bias=qkv_bias)
        self.v = nn.Linear(embed_dim, self.attention_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        if self.cross:
            self.g = nn.Sequential(nn.Linear(embed_dim, self.attention_dim,bias=qkv_bias), nn.SiLU())
            self.proj = nn.Linear(self.attention_dim + self.num_head*self.attn_num, dim, bias=proj_bias)
        else:
            self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_pytorch = (args.mat_impl == 'pytorch')
        self.mf_fused = MF_FusedForwardOps()

    def clamp_max_offset(self, max_offset, H, W):
        max_offset_x, max_offset_y = max_offset.chunk(2, dim=-1) # to avoid inplace operation

        # for ONNX support
        min_x = torch.tensor(self.win_r[0], dtype=max_offset.dtype, device=max_offset.device)
        max_x = torch.tensor(W - 1 - self.win_r[0] - 1e-3, dtype=max_offset.dtype, device=max_offset.device)
        min_y = torch.tensor(self.win_r[1], dtype=max_offset.dtype, device=max_offset.device)
        max_y = torch.tensor(H - 1 - self.win_r[1] - 1e-3, dtype=max_offset.dtype, device=max_offset.device)

        max_offset_x = torch.clamp(max_offset_x, min=min_x, max=max_x)
        max_offset_y = torch.clamp(max_offset_y, min=min_y, max=max_y)

        ## max_offset_x = max_offset_x.clamp(min=self.win_r[0], max=W-1-self.win_r[0]-1e-3)
        ## max_offset_y = max_offset_y.clamp(min=self.win_r[1], max=H-1-self.win_r[1]-1e-3)
        return torch.cat((max_offset_x, max_offset_y), dim=-1).contiguous()

    def forward(self, x, max_offset, noc_mask=None): # offset: [B, N, h, 2]
        B, H, W, _ = x.shape
        N = H*W
        assert (2*self.win_r[1] + 2 <= H) and (2*self.win_r[0] + 2 <= W)
        x = x.view(B, N, -1).contiguous()

        if self.cross:
            ref_, tgt_ = x.chunk(2, dim=0) # split along batch dimension 
            ref = torch.cat((ref_, tgt_), dim=0) # order
            tgt = torch.cat((tgt_, ref_), dim=0) # reverse order
            g = self.g(ref)
        else: # self-attn
            if self.noc_embed:
                x = torch.cat((x, noc_mask.view(B, N, -1)), dim=-1).contiguous()
            ref, tgt = x, x
        q, k, v = self.q(ref), self.k(tgt), self.v(tgt)

        ## non-parameter modules
        max_offset = self.clamp_max_offset(max_offset, H, W)

        if self.use_pytorch:
            m_id = torch.floor(max_offset).to(torch.int32) # [B, N, h, 2]
            bilinear_weight = compute_bilinear_weights(max_offset)

            attn, indices_gather = compute_match_attention(q.view(B, N, self.num_head, -1), k.view(B, N, self.num_head, -1), m_id, self.win_r, H, W)
            attn = attn * self.scale

            attn = compute_bilinear_softmax(attn, bilinear_weight, self.win_r)
            attn = self.attn_drop(attn)

            x = attention_aggregate(v.view(B, N, self.num_head, -1), attn, indices_gather, self.win_r)
        else:
            x, attn = self.mf_fused(max_offset, q, k, v, H, W, self.win_r, self.attn_num, attn_type='l1_norm', scale=self.scale)

        if self.cross:
            x = g * x # gate
            attn = attn.view(B, N, -1).contiguous()
            x = torch.cat((x, attn), dim=-1).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x.view(B, H, W, -1).contiguous()


class MatchAttentionLayer(nn.Module):
    r"""MatchAttention layer with interleaved self-MatchAttention, cross-MatchAttention, and ConvGLU
    """

    def __init__(self, args, dim, win_r,
                 num_head=8, head_dim=32, mlp=ConvGLU, mlp_ratio=2, field_dim=2,
                 norm_layer=nn.LayerNorm, drop=0., drop_path=0.):
        super().__init__()
        self.num_head = num_head
        self.field_dim = field_dim

        self.match_attention_self = MatchAttention(args, dim + self.field_dim + self.num_head*2, [win_r, win_r], num_head=num_head, head_dim=head_dim, noc_embed=True)
        self.norm0 = norm_layer(dim + self.field_dim + self.num_head*2)

        self.match_attention_cross = MatchAttention(args, dim + self.field_dim, [win_r, win_r], num_head=num_head, head_dim=head_dim, cross=True)
        self.norm1 = norm_layer(dim + self.field_dim)

        self.mlp = mlp(dim=dim, mlp_ratio=mlp_ratio, drop=drop)
        self.norm2 = norm_layer(dim)

        self.field_scale = nn.Parameter(0.1*torch.ones(1, 1, 1, 2))

        self.drop_path0 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def consistency_mask(self, field, A=2):
        offset = field + init_coords(field) # [B, H, W, 2]
        field_ref_, field_tgt_ = field.chunk(2, dim=0)
        field_ref = torch.cat((field_ref_, field_tgt_), dim=0) # order
        field_tgt = torch.cat((field_tgt_, field_ref_), dim=0) # reverse order
        field_tgt_to_ref = bilinear_sample_by_offset(field_tgt.permute(0, 3, 1, 2).contiguous(), offset).permute(0, 2, 3, 1).contiguous()
        field_diff = torch.abs(field_ref + field_tgt_to_ref).sum(dim=-1, keepdim=True) # ref and tgt flow has different sign
        noc_mask = (field_diff < A).to(field_diff.dtype)
        return noc_mask

    def forward(self, x, self_rpos, field, stereo=True): # self_rpos [B, H, W, h*2], field [B, H, W, 2]

        field_out = {}
        B, H, W, C = x.shape

        noc_mask = self.consistency_mask(field.detach())

        x = torch.cat((x, field*self.field_scale.to(field.dtype), self_rpos), dim=-1).contiguous()

        coords_0 = init_coords(field).repeat(1, 1, 1, self.num_head)
        self_offset = self_rpos + coords_0
        self_offset = self_offset.view(B, H*W, self.num_head, 2).contiguous()

        x = x + self.drop_path0(self.match_attention_self(self.norm0(x), self_offset, noc_mask))

        self_rpos = x[..., -(self.num_head*2):].contiguous() # [B, H, W, h*2]
        x = x[..., :-(self.num_head*2)].contiguous()

        if stereo: x[..., -1] = 0
        field = x[..., -self.field_dim:].contiguous() / self.field_scale.to(field.dtype)
        field_out['self'] = field.clone()

        offset = field.repeat(1, 1, 1, self.num_head).contiguous() + coords_0 # [B, H, W, h*2]
        offset = offset.view(B, H*W, self.num_head, 2).contiguous()

        x = x + self.drop_path1(self.match_attention_cross(self.norm1(x), offset))

        if stereo: x[..., -1] = 0
        field = x[..., -self.field_dim:].contiguous() / self.field_scale.to(field.dtype)
        field_out['cross'] = field.clone()

        x = x[..., :-self.field_dim].contiguous() # No field feature in MLP

        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x, self_rpos, field, field_out


class MatchAttentionBlock(nn.Module):
    r"""MatchAttention block with multiple match-attention layers
    """

    def __init__(self, args, dim, win_r=2,
                 num_layer=6, num_head=8, head_dim=32,
                 mlp=ConvGLU, mlp_ratio=2, field_dim=2,
                 norm_layer=LayerNormWithoutBias,
                 drop=0., dp_rates=[0.]):

        super().__init__()
        self.num_head = num_head

        self.layers = nn.ModuleList()
        for i in range(num_layer):
            layer = MatchAttentionLayer(args, dim, win_r=win_r, num_head=num_head, head_dim=head_dim,
                                        mlp=mlp, mlp_ratio=mlp_ratio, field_dim=field_dim,
                                        norm_layer=norm_layer, drop=drop, drop_path=dp_rates[i])
            self.layers.append(layer)

    def forward(self, x, self_rpos, field, stereo=True):
        fields = []
        B, H, W, C = x.shape
        self_rpos = self_rpos.repeat(1, 1, 1, self.num_head) # [B, H, W, 2] -> [B, H, W, h*2]

        for layer in self.layers:

            x, self_rpos, field, field_out = layer(x, self_rpos, field, stereo)
            fields.append(field_out)

        self_rpos = self_rpos.view(B, H, W, self.num_head, 2).mean(dim=-2, keepdim=False)

        return x, self_rpos, field, fields
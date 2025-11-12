import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
from models.common import UpConv
from models.convformer import convformer
from models.attention_blocks import MatchAttentionBlock
from models.cost_volume import GlobalCorrelation

class MatchStereo(nn.Module):
    def __init__(self, args,
                 refine_win_rs=[2, 2, 1, 1], # refine window radius at 1/32, 1/16, 1/8, 1/4
                 refine_nums=[8, 8, 8, 2],
                 num_heads=[4, 4, 4, 4],
                 mlp_ratios=[2, 2, 2, 2],
                 drop_path=0.):
        super().__init__()
        self.refine_nums = refine_nums

        self.encoder = convformer(args.variant)
        self.channels = self.encoder.dims[::-1] # resolution low to high
        self.num_heads = num_heads
        self.head_dims = [c//h for c, h in zip(self.channels, self.num_heads)]

        self.factor = 2
        self.factor_last = 2**(len(self.channels) - len(refine_nums) + 2)
        
        self.field_dim = 2 # 2(flow)

        self.up_decoders = nn.ModuleList()
        self.up_masks = nn.ModuleList()
        for i in range(len(self.channels)):
            if i > 0:
                self.up_decoders.append(UpConv(self.channels[i-1], self.channels[i]))
                self.up_masks.append(
                    nn.Sequential(
                    nn.Conv2d(self.channels[i-1], self.channels[i-1], 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.channels[i-1], (self.factor**2)*9, 1, padding=0))
                )
            else:
                self.up_decoders.append(nn.Identity())
                self.up_masks.append(nn.Identity())

        self.up_masks.append(
            nn.Sequential(
            nn.Conv2d(self.channels[-1], self.channels[-1]*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels[-1]*2, (self.factor_last**2)*9, 1, padding=0)))

        dp_rates = [x.item() for x in torch.linspace(0, drop_path, sum(refine_nums))]
        # MatchAttention
        self.match_attentions = nn.ModuleList()
        for i in range(len(refine_nums)):
            self.match_attentions.append(
                MatchAttentionBlock(args, self.channels[i], win_r=refine_win_rs[i], 
                                    num_layer=refine_nums[i], num_head=self.num_heads[i], head_dim=self.head_dims[i], 
                                    mlp_ratio=mlp_ratios[i], field_dim=self.field_dim, 
                                    dp_rates=dp_rates[sum(refine_nums[:i]):sum(refine_nums[:i+1])])
            )

        self.init_correlation_volume = GlobalCorrelation(self.channels[0])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def upsample_field(self, field, mask, factor):
        ''' Upsample field [H/factor, W/factor, D] -> [H, W, D] using convex combination '''
        B, H, W, D = field.shape
        field = field.permute(0, 3, 1, 2)
        mask = mask.view(B, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2).to(mask.dtype)
        up_flow = F.unfold(field*factor, [3,3], padding=1)
        up_flow = up_flow.view(B, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2).to(mask.dtype) # [B, D, 9, factor, factor, H, W]
        up_flow = up_flow.permute(0, 4, 2, 5, 3, 1)
        return up_flow.reshape(B, factor*H, factor*W, D).contiguous()

    def forward(self, img0, img1, stereo=True, init_flow=None):
        ''' Estimate optical flow/disparity between pair of frames, output bi-directional flow/disparity '''
        field_all = []

        img0 = (2 * (img0 / 255.0) - 1.0).contiguous()
        img1 = (2 * (img1 / 255.0) - 1.0).contiguous()

        x = torch.cat((img0, img1), dim=0) # cat in batch dim

        features = self.encoder(x) # [B*2, H, W, C]
        features = features[::-1] # reverse 1/32, 1/16, 1/8, 1/4
        
        for i in range(len(features)): # 1/32, 1/16, 1/8, 1/4
            if i==0:
                if init_flow is None:
                    init_flow, init_cv = self.init_correlation_volume(features[i], stereo=stereo)
                else:
                    init_cv = None

                field = init_flow.clone() # [B, H, W, 2]
                self_rpos = torch.zeros_like(field)
            else:
                features[i] = self.up_decoders[i](features[i-1], features[i])
                up_mask = self.up_masks[i](features[i-1].permute(0, 3, 1, 2)) # [B, C, H, W]
                self_rpos = self.upsample_field(self_rpos, up_mask, self.factor)
                field = self.upsample_field(field, up_mask, self.factor)
                field_all.append({'self':field})

            features[i], self_rpos, field, fields = self.match_attentions[i](features[i], self_rpos, field, stereo=stereo)
            field_all.extend(fields)

        if self.training:
            B = field.shape[0]
            field_up = self.upsample_field(field[:B//2], self.up_masks[-1](features[-1][:B//2].permute(0, 3, 1, 2)), self.factor_last)
            field_up = torch.cat((field_up, field_up), dim=0) # dummy output
        else:
            field_up = self.upsample_field(field, self.up_masks[-1](features[-1].permute(0, 3, 1, 2)), self.factor_last)

        return {
            'init_flow': init_flow,
            'init_cv': init_cv,
            'field_all': field_all,
            'field_up': field_up,
            'self_rpos': self_rpos,
        }
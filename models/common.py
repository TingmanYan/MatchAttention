import torch
import torch.nn as nn
import torch.nn.functional as F

class UpConv(nn.Module):
    r"""Upsample using transposed conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x1, x2, use_up=True):
        x1 = x1.permute(0, 3, 1, 2).contiguous()
        x2 = x2.permute(0, 3, 1, 2).contiguous()
        if use_up:
            x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        out = self.conv(x)
        return out.permute(0, 2, 3, 1).contiguous() # [B, H, W, C]

class ConvGLU(nn.Module):
    '''
    Convolutional GLU, referenced from TransNeXt
    '''
    def __init__(self, dim, mlp_ratio=2, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x): # [B, H, W, C]
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
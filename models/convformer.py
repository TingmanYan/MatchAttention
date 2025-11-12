from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple
class LayerNormGeneral(nn.Module):
    r""" General LayerNorm for different situations.

    Args:
        affine_shape (int, list or tuple): The shape of affine weight and bias.
            Usually the affine_shape=C, but in some implementation, like torch.nn.LayerNorm,
            the affine_shape is the same as normalized_dim by default. 
            To adapt to different situations, we offer this argument here.
        normalized_dim (tuple or list): Which dims to compute mean and variance. 
        scale (bool): Flag indicates whether to use scale or not.
        bias (bool): Flag indicates whether to use scale or not.

        We give several examples to show how to specify the arguments.

        LayerNorm (https://arxiv.org/abs/1607.06450):
            For input shape of (B, *, C) like (B, N, C) or (B, H, W, C),
                affine_shape=C, normalized_dim=(-1, ), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, ), scale=True, bias=True.

        Modified LayerNorm (https://arxiv.org/abs/2111.11418)
            that is idental to partial(torch.nn.GroupNorm, num_groups=1):
            For input shape of (B, N, C),
                affine_shape=C, normalized_dim=(1, 2), scale=True, bias=True;
            For input shape of (B, H, W, C),
                affine_shape=C, normalized_dim=(1, 2, 3), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, 2, 3), scale=True, bias=True.

        For the several metaformer baslines,
            IdentityFormer, RandFormer and PoolFormerV2 utilize Modified LayerNorm without bias (bias=False);
            ConvFormer and CAFormer utilizes LayerNorm without bias (bias=False).
    """

    def __init__(self, affine_shape=None, normalized_dim=(-1, ), scale=True,
                 bias=False, eps=1e-6):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x



def stem(in_chs, out_chs, act_layer=nn.GELU):
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
        ## nn.BatchNorm2d(out_chs // 2),
        nn.InstanceNorm2d(out_chs // 2),
        act_layer(),
        nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
        ## nn.BatchNorm2d(out_chs),
        nn.InstanceNorm2d(out_chs),
        act_layer(),
    )

class Downsampling(nn.Module):
    """
    Downsampling implemented by a layer of convolution.
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=2, padding=1,
                 pre_norm=LayerNormGeneral, post_norm=None, pre_permute=True):
        super().__init__()
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.post_norm = post_norm(
            out_channels) if post_norm else nn.Identity()

    def forward(self, x):
        x = self.pre_norm(x)
        if self.pre_permute:
            x = x.permute(0, 3, 1, 2).contiguous() # if take [B, H, W, C] as input, permute it to [B, C, H, W]
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W] -> [B, H, W, C]
        x = self.post_norm(x)
        return x


class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """

    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class LayerNormWithoutBias(nn.Module):
    """
    Equal to partial(LayerNormGeneral, bias=False) but faster, 
    because it directly utilizes otpimized F.layer_norm
    """

    def __init__(self, normalized_shape, eps=1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.bias = None
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(self, dim, expansion_ratio=2,
                 act1_layer=nn.GELU, act2_layer=nn.Identity,
                 bias=False, kernel_size=3, padding=1,
                 **kwargs, ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=padding, groups=med_channels, bias=bias)  # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act2(x)
        x = self.pwconv2(x)
        return x

class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """

    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=nn.GELU, drop=0., bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MetaFormerBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    """

    def __init__(self, dim,
                 token_mixer=nn.Identity, mlp=Mlp, mlp_ratio=4,
                 norm_layer=nn.LayerNorm, drop=0., drop_path=0.,
                 layer_scale_init_value=None, res_scale_init_value=None
                 ):

        super().__init__()

        self.token_mixer = token_mixer(dim, drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, mlp_ratio=mlp_ratio, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.token_mixer(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class MetaFormer(nn.Module):
    r""" MetaFormer
        A PyTorch impl of : `MetaFormer Baselines for Vision`  -
          https://arxiv.org/abs/2210.13452

    Args:
        in_chans (int): Number of input image channels. Default: 3.
        num_classes (int): Number of classes for classification head. Default: 1000.
        depths (list or tuple): Number of blocks at each stage. Default: [2, 2, 6, 2].
        dims (int): Feature dimension at each stage. Default: [64, 128, 320, 512].
        downsample_layers: (list or tuple): Downsampling layers before each stage.
        token_mixers (list, tuple or token_fcn): Token mixer for each stage. Default: nn.Identity.
        mlps (list, tuple or mlp_fcn): Mlp for each stage. Default: Mlp.
        norm_layers (list, tuple or norm_fcn): Norm layers for each stage. Default: partial(LayerNormGeneral, eps=1e-6, bias=False).
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_values (list, tuple, float or None): Init value for Layer Scale. Default: None.
            None means not use the layer scale. Form: https://arxiv.org/abs/2103.17239.
        res_scale_init_values (list, tuple, float or None): Init value for Layer Scale. Default: [None, None, 1.0, 1.0].
            None means not use the layer scale. From: https://arxiv.org/abs/2110.09456.
        head_fn: classification head. Default: nn.Linear.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[2, 2, 6, 2],
                 dims=[64, 128, 320, 512],
                 downsample_layers=[stem] + [Downsampling]*3,
                 token_mixers=nn.Identity,
                 mlps=Mlp, mlp_ratio=4,
                 norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
                 drop_path_rate=0.,
                 layer_scale_init_values=None,
                 res_scale_init_values=[None, None, 1.0, 1.0],
                 head_fn=nn.Linear,
                 **kwargs,
                 ):
        super().__init__()
        self.num_classes = num_classes

        if not isinstance(depths, (list, tuple)):
            depths = [depths]  # it means the model has only one stage
        if not isinstance(dims, (list, tuple)):
            dims = [dims]

        self.dims = dims
        self.depths = depths

        num_stage = len(depths)
        self.num_stage = num_stage

        down_dims = [in_chans] + dims
        self.downsample_layers = nn.ModuleList([downsample_layers[i](down_dims[i], down_dims[i+1]) for i in range(num_stage)])

        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * num_stage
        self.token_mixers = token_mixers

        if not isinstance(mlps, (list, tuple)):
            mlps = [mlps] * num_stage

        if not isinstance(norm_layers, (list, tuple)):
            norm_layers = [norm_layers] * num_stage

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        if not isinstance(layer_scale_init_values, (list, tuple)):
            layer_scale_init_values = [layer_scale_init_values] * num_stage
        if not isinstance(res_scale_init_values, (list, tuple)):
            res_scale_init_values = [res_scale_init_values] * num_stage

        self.stages = nn.ModuleList()  # each stage consists of multiple metaformer blocks
        cur = 0
        for i in range(num_stage):
            stage = nn.ModuleList(
                [MetaFormerBlock(dim=dims[i], token_mixer=token_mixers[i],
                                  mlp=mlps[i], mlp_ratio=mlp_ratio, norm_layer=norm_layers[i],
                                  drop_path=dp_rates[cur + j],
                                  layer_scale_init_value=layer_scale_init_values[i],
                                  res_scale_init_value=res_scale_init_values[i],
                                  ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.head = head_fn(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        outs = []
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            if i==0: x = x.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W] -> [B, H, W, C]
            for j in range(self.depths[i]):
                x= self.stages[i][j](x)
            outs.append(x) # [B, H, W, C]
        return outs

def convformer(variant='tiny'):
    if variant == 'tiny':
        model = convformer_t()

    elif variant == 'small':
        model = convformer_s()
        
    elif variant == 'base':
        model = convformer_b()

    elif variant == 'large':
        model = convformer_l()

    else:
        raise NotImplementedError

    return model

@register_model
def convformer_t(**kwargs):
    model = MetaFormer(
        depths=[2, 2, 6, 2],
        dims=[32, 64, 128, 160],
        mlps=Mlp, mlp_ratio=2,
        token_mixers=[SepConv, SepConv, SepConv, SepConv],
        head_fn=nn.Linear,
        **kwargs)
    return model

@register_model
def convformer_s(**kwargs):
    model = MetaFormer(
        depths=[2, 2, 6, 2],
        dims=[64, 128, 160, 320],
        mlps=Mlp, mlp_ratio=2,
        token_mixers=[SepConv, SepConv, SepConv, SepConv],
        head_fn=nn.Linear,
        **kwargs)
    return model

@register_model
def convformer_b(**kwargs):
    model = MetaFormer(
        depths=[2, 2, 6, 2],
        dims=[128, 256, 320, 512],
        mlps=Mlp, mlp_ratio=2,
        token_mixers=[SepConv, SepConv, SepConv, SepConv],
        head_fn=nn.Linear,
        **kwargs)
    return model

@register_model
def convformer_l(**kwargs):
    model = MetaFormer(
        depths=[2, 2, 6, 2],
        dims=[256, 384, 512, 768],
        mlps=Mlp, mlp_ratio=2,
        token_mixers=[SepConv, SepConv, SepConv, SepConv],
        head_fn=nn.Linear,
        **kwargs)
    return model
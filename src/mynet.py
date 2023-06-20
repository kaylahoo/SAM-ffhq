"""
BiFormer impl.
author: ZHU Lei
github: https://github.com/rayleizhu
email: ray.leizhu@outlook.com
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import math
from collections import OrderedDict
from functools import partial
from typing import Optional, Union
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models import register_model
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg

from src.bra_legacy import BiLevelRoutingAttention

from src._common import Attention, AttentionLePE, DWConv
from src import attention


# from torchinfo import summary
# from positional_encodings import PositionalEncodingPermute2D, Summer
# from siren_pytorch import SirenNet

class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(True)
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        out = self.conv(x)
        return out


# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(True)
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        out = self.deconv(x)
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim, h, w):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x):
        # print(x.shape)
        # B, N, C = x.shape
        # if spatial_size is None:
        #     a = b = int(math.sqrt(N))
        # else:
        #     a, b = spatial_size

        # x = x.view(B, a, b, C)
        # print(x.shape)
        x = x.to(torch.float32)
        # print(x.shape)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        # print("傅里叶变换后的形状",x.shape)
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(self.h, self.h), dim=(1, 2), norm='ortho')
        # print(x.shape)

        # x = x.reshape(B, N, C)

        return x


class Block_spect(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = SpectralGatingNetwork(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # print("进入block的形状",x.shape)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x


class Block_att(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=-1,
                 num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='ada_avgpool',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                 mlp_ratio=4, mlp_dwconv=False,
                 side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=False):
        super().__init__()
        qk_dim = qk_dim or dim

        # modules
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  # important to avoid attention collapsing
        if topk > 0:
            self.attn = BiLevelRoutingAttention(dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                                                qk_scale=qk_scale, kv_per_win=kv_per_win,
                                                kv_downsample_ratio=kv_downsample_ratio,
                                                kv_downsample_kernel=kv_downsample_kernel,
                                                kv_downsample_mode=kv_downsample_mode,
                                                topk=topk, param_attention=param_attention, param_routing=param_routing,
                                                diff_routing=diff_routing, soft_routing=soft_routing,
                                                side_dwconv=side_dwconv,
                                                auto_pad=auto_pad)
        elif topk == -1:
            self.attn = Attention(dim=dim)
        elif topk == -2:
            self.attn = AttentionLePE(dim=dim, side_dwconv=side_dwconv)
        elif topk == 0:
            self.attn = nn.Sequential(Rearrange('n h w c -> n c h w'),  # compatiability
                                      nn.Conv2d(dim, dim, 1),  # pseudo qkv linear
                                      nn.Conv2d(dim, dim, 5, padding=2, groups=dim),  # pseudo attention
                                      nn.Conv2d(dim, dim, 1),  # pseudo out linear
                                      Rearrange('n c h w -> n h w c')
                                      )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio * dim)),
                                 DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio * dim), dim)
                                 )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # tricks: layer scale & pre_norm/post_norm
        if layer_scale_init_value > 0:
            self.use_layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.use_layer_scale = False
        self.pre_norm = pre_norm

    def forward(self, x):
        """
        x: NCHW tensor
        """
        # conv pos embedding
        x = x + self.pos_embed(x)
        # permute to NHWC tensor for attention & mlp
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        # attention & mlp
        if self.pre_norm:
            if self.use_layer_scale:
                x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))  # (N, H, W, C)
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))  # (N, H, W, C)
            else:
                x = x + self.drop_path(self.attn(self.norm1(x)))  # (N, H, W, C)
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # (N, H, W, C)
        else:  # https://kexue.fm/archives/9009
            if self.use_layer_scale:
                x = self.norm1(x + self.drop_path(self.gamma1 * self.attn(x)))  # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.gamma2 * self.mlp(x)))  # (N, H, W, C)
            else:
                x = self.norm1(x + self.drop_path(self.attn(x)))  # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.mlp(x)))  # (N, H, W, C)

        # permute back
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x


class BiFormer(nn.Module):
    def __init__(self, depth=[4, 4, 18, 4, 18, 4, 4], in_chans=3, num_classes=1000, embed_dim=[64, 128, 320, 512],
                 head_dim=64, qk_scale=None, representation_size=None,
                 drop_path_rate=0., drop_rate=0.,
                 use_checkpoint_stages=[],
                 ########
                 n_win=7,
                 kv_downsample_mode='ada_avgpool',
                 kv_per_wins=[2, 2, -1, -1],
                 topks=[8, 8, -1, -1],
                 side_dwconv=5,
                 layer_scale_init_value=-1,
                 qk_dims=[None, None, None, None],
                 param_routing=False, diff_routing=False, soft_routing=False,
                 pre_norm=True,
                 pe=None,
                 pe_stages=[0],
                 before_attn_dwconv=3,
                 auto_pad=False,
                 # -----------------------
                 kv_downsample_kernels=[4, 2, 1, 1, 1, 2, 4],
                 kv_downsample_ratios=[4, 2, 1, 1, 1, 2, 4],  # -> kv_per_win = [2, 2, 2, 1]
                 mlp_ratios=[4, 4, 4, 4],
                 param_attention='qkvo',
                 mlp_dwconv=False):
        """
        Args:
            depth (list): depth of each stage
            img_size (int, tuple): input image size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (list): embedding dimension of each stage
            head_dim (int): head dimension
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer (nn.Module): normalization layer
            conv_stem (bool): whether use overlapped patch stem
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.start = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.GELU(),
        )
        self.spec0 = nn.Sequential(
            *[Block_spect(dim=32, h=128, w=65) for i in range(2)]
        )

        self.down64 = Downsample(in_channel=32, out_channel=64)

        self.spec1 = nn.Sequential(
            *[Block_spect(dim=64, h=128, w=65) for i in range(2)]
        )


        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        nheads = [dim // head_dim for dim in qk_dims]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0
        for i in range(len(depth)):
            stage = nn.Sequential(
                *[Block_att(dim=embed_dim[i], drop_path=dp_rates[cur + j],
                            layer_scale_init_value=layer_scale_init_value,
                            topk=topks[i],
                            num_heads=nheads[i],
                            n_win=n_win,
                            qk_dim=qk_dims[i],
                            qk_scale=qk_scale,
                            kv_per_win=kv_per_wins[i],
                            kv_downsample_ratio=kv_downsample_ratios[i],
                            kv_downsample_kernel=kv_downsample_kernels[i],
                            kv_downsample_mode=kv_downsample_mode,
                            param_attention=param_attention,
                            param_routing=param_routing,
                            diff_routing=diff_routing,
                            soft_routing=soft_routing,
                            mlp_ratio=mlp_ratios[i],
                            mlp_dwconv=mlp_dwconv,
                            side_dwconv=side_dwconv,
                            before_attn_dwconv=before_attn_dwconv,
                            pre_norm=pre_norm,
                            auto_pad=auto_pad) for j in range(depth[i])],
            )
            if i in use_checkpoint_stages:
                stage = checkpoint_wrapper(stage)
            self.stages.append(stage)
            cur += depth[i]

        self.down32 = Downsample(in_channel=64, out_channel=128)
        self.down16 = Downsample(in_channel=128, out_channel=256)
        self.down8 = Downsample(in_channel=256, out_channel=512)
        self.up16 = Upsample(512, 256)
        self.up32 = Upsample(512, 128)
        self.up64 = Upsample(256, 64)
        self.up128 = Upsample(128, 32)

        self.out = nn.Sequential(

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=7, padding=0)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def forward(self, x, masks):
        gt = x
        x = torch.cat((x, masks), dim=1)  # torch.Size([1, 4, 256, 256])
        x = self.start(x)  # torch.Size([1, 32, 128, 128])

        spect0 = self.spec0(x)  # torch.Size([1, 32, 128, 128])

        down64 = self.down64(spect0)  # torch.Size([1, 64, 64, 64])


        stage0 = self.stages[0](down64)  # torch.Size([1, 64, 64, 64])
        down32 = self.down32(stage0)  # torch.Size([1, 128, 32, 32])

        stage1 = self.stages[1](down32)  # torch.Size([1, 128, 32, 32])
        down16 = self.down16(stage1)  # torch.Size([1, 256, 16, 16])

        stage2 = self.stages[2](down16)  # torch.Size([1, 256, 16, 16])
        down8 = self.down8(stage2)  # torch.Size([1, 512, 8, 8])

        #  Bottleneck
        stage3 = self.stages[3](down8)  # torch.Size([1, 512, 8, 8])

        # Decoder
        up16 = self.up16(stage3)  # torch.Size([1, 256, 16, 16])
        deconv0 = torch.cat([up16, stage2], 1)  # torch.Size([1, 512, 16, 16])
        deconv0 = self.stages[4](deconv0)  # torch.Size([1, 512, 16, 16])

        up32 = self.up32(deconv0)  # torch.Size([1, 128, 32, 32])
        deconv1 = torch.cat([up32, stage1], 1)  # torch.Size([1, 256, 32, 32])
        deconv1 = self.stages[5](deconv1)  # torch.Size([1, 256, 32, 32])

        #
        up64 = self.up64(deconv1)  # torch.Size([1, 64, 64, 64])
        deconv2 = torch.cat([up64, stage0], 1)  # torch.Size([1, 128, 64, 64])
        deconv2 = self.stages[6](deconv2)  # torch.Size([1, 128, 64, 64])

        up128 = self.up128(deconv2) #torch.Size([1, 64, 128, 128])
        # print(up128.shape)
        deconv3 = torch.cat([up128, spect0], 1)
        deconv3 = self.spec1(deconv3)#torch.Size([1, 128, 128, 128])

        # print(deconv4.shape)
        #
        output = self.out(deconv3)
        # print(output.shape)
        output = (torch.tanh(output) + 1) / 2
        # print(output.shape)
        # image_stage1 = (x * masks) + (gt * (1 - masks))
        # image_stage2 = self.middle(torch.cat((image_stage1, masks), dim=1))
        # image_stage3 = (image_stage2 * masks) + (gt * (1 - masks))
        # output = self.refine(torch.cat((image_stage3, masks), dim=1))
        # output = (torch.tanh(output) + 1) / 2
        return output


#################### model variants #######################


model_urls = {
    "biformer_tiny_in1k": "https://api.onedrive.com/v1.0/shares/s!AkBbczdRlZvChHEOoGkgwgQzEDlM/root/content",
    "biformer_small_in1k": "https://api.onedrive.com/v1.0/shares/s!AkBbczdRlZvChHDyM-x9KWRBZ832/root/content",
    "biformer_base_in1k": "https://api.onedrive.com/v1.0/shares/s!AkBbczdRlZvChHI_XPhoadjaNxtO/root/content",
}


@register_model
def biformer_small(pretrained=False, pretrained_cfg=None,
                   pretrained_cfg_overlay=None, **kwargs):
    model = BiFormer(
        depth=[4, 4, 4, 8, 4, 4, 4],
        embed_dim=[64, 128, 256, 512, 512, 256, 128], mlp_ratios=[3, 3, 3, 3, 3, 3, 3],
        # ------------------------------
        n_win=8,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1, -1, -1, -1],
        topks=[1, 4, 16, -2, 16, 4, 1],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[64, 128, 256, 512, 512, 256, 128],
        head_dim=32,
        param_routing=False, diff_routing=False, soft_routing=False,
        pre_norm=True,
        pe=None,
        # -------------------------------
        **kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        model_key = 'biformer_small_in1k'
        url = model_urls[model_key]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True,
                                                        file_name=f"{model_key}.pth")
        model.load_state_dict(checkpoint["model"])

    return model


@register_model
def biformer_base(pretrained=False, pretrained_cfg=None,
                  pretrained_cfg_overlay=None, **kwargs):
    model = BiFormer(
        depth=[4, 4, 18, 4],
        embed_dim=[96, 192, 384, 768], mlp_ratios=[3, 3, 3, 3],
        # use_checkpoint_stages=[0, 1, 2, 3],
        use_checkpoint_stages=[],
        # ------------------------------
        n_win=8,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[96, 192, 384, 768],
        head_dim=32,
        param_routing=False, diff_routing=False, soft_routing=False,
        pre_norm=True,
        pe=None,
        # -------------------------------
        **kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        model_key = 'biformer_base_in1k'
        url = model_urls[model_key]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True,
                                                        file_name=f"{model_key}.pth")
        model.load_state_dict(checkpoint["model"])

    return model


if __name__ == '__main__':
    net = biformer_small()
    x = torch.randn((1, 3, 256, 256))
    x = net(x, masks=torch.randn((1, 1, 256, 256)))
    # total_params = sum(p.numel() for p in net.parameters())
    # print(f"Total parameters: {total_params}")
from collections import OrderedDict
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath


NEG_INF = -1000000

class Mlp(nn.Module):
    r"""2-layer MLP"""
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

class DynamicPosBias(nn.Module):
    r"""DPB module
    
    Use a MLP to predict position bias used in attention.
    """
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )
    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases) # 2Wh-1 * 2Ww-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos



class Attention(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias
        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)


        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Gh*Gw, Gh*Gw) or None
        """
        group_size = (H, W)
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous()) # (num_windows*B, N, N), N = Gh*Gw

        if self.position_bias:
            # generate mother-set
            position_bias_h = torch.arange(1 - group_size[0], group_size[0], device=attn.device)
            position_bias_w = torch.arange(1 - group_size[1], group_size[1], device=attn.device)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Gh-1, 2W2-1
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(group_size[0], device=attn.device)
            coords_w = torch.arange(group_size[1], device=attn.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Gh, Gw
            coords_flatten = torch.flatten(coords, 1)  # 2, Gh*Gw
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Gh*Gw, Gh*Gw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Gh*Gw, Gh*Gw, 2
            relative_coords[:, :, 0] += group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Gh*Gw, Gh*Gw

            pos = self.pos(biases) # 2Gh-1 * 2Gw-1, heads
            # select position bias
            relative_position_bias = pos[relative_position_index.view(-1)].view(
                group_size[0] * group_size[1], group_size[0] * group_size[1], -1)  # Gh*Gw,Gh*Gw,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Gh*Gw, Gh*Gw
            attn = attn + relative_position_bias.unsqueeze(0)


        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).contiguous().reshape(B_, N, C)

        return x



class CrossFormerBlock(nn.Module):
    r""" CrossFormer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        group_size (int): Group size.
        interval (int): Interval for LDA.
        lsda_flag (int): use SDA or LDA, 0 for SDA and 1 for LDA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        num_patch_size
        impl_type (str): 
        use_extra_conv (bool): Extra convolution layer. Default: True
    """

    def __init__(self, dim=64, input_resolution=[40,40], num_heads=2, group_size=20, interval=8, lsda_flag=0,
                 mlp_ratio=1., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_patch_size=1,
                 pad_type=0, use_extra_conv=True, use_cpe=False, no_mask=False, adaptive_interval=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.group_size = group_size
        self.interval = interval
        self.lsda_flag = lsda_flag
        self.mlp_ratio = mlp_ratio
        self.num_patch_size = num_patch_size
        self.pad_type = pad_type
        self.use_extra_conv = use_extra_conv
        self.use_cpe = use_cpe
        self.adaptive_interval = adaptive_interval

        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            position_bias=(not use_cpe))

        if self.use_cpe:
            self.cpe = nn.Conv2d(in_channels=input_resolution[0], out_channels=input_resolution[0], kernel_size=3, padding=1, groups=input_resolution[0])
            self.norm_cpe = norm_layer(dim)

        # if adaptive_interval:
        #     self.interval = int(np.ceil(self.input_resolution[0] / self.group_size))

        if self.use_extra_conv:
            self.ex_kernel = [3, 3]
            padding = (self.ex_kernel[0] - 1) // 2
            self.ex_conv = nn.Conv2d(dim, dim, self.ex_kernel, padding=padding, groups=dim)
            self.ex_ln = norm_layer(dim)
        #
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim, elementwise_affine=True)
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        # H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size %d, %d, %d" % (L, H, W)

        if min(H, W) <= self.group_size:
            # if window size is larger than input resolution, we don't partition windows
            self.lsda_flag = 0
            # group_size = min(H, W)
            group_size = max(H, W)
        else:
            group_size = self.group_size

        if self.adaptive_interval:
            self.interval = math.ceil(max(H, W) / group_size)

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.use_cpe:
            x = x + self.norm_cpe(self.cpe(x))

        # padding
        size_div = self.interval * group_size if self.lsda_flag == 1 else group_size
        pad_l = pad_t = 0
        pad_r = (size_div - W % size_div) % size_div
        pad_b = (size_div - H % size_div) % size_div
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        if pad_b > 0:
            mask[:, -pad_b:, :, :] = -1
        if pad_r > 0:
            mask[:, :, -pad_r:, :] = -1

        # group embeddings and generate attn_mask
        if self.lsda_flag == 0: # SDA
            G = Gh = Gw = group_size
            x = x.reshape(B, Hp // G, G, Wp // G, G, C).permute(0, 1, 3, 2, 4, 5).contiguous()
            x = x.reshape(B * Hp * Wp // G**2, G**2, C)
            nG = Hp * Wp // G**2
            # attn_mask
            if pad_r > 0 or pad_b > 0:
                mask = mask.reshape(1, Hp // G, G, Wp // G, G, 1).permute(0, 1, 3, 2, 4, 5).contiguous()
                mask = mask.reshape(nG, 1, G * G)
                attn_mask = torch.zeros((nG, G * G, G * G), device=x.device)
                attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
            else:
                attn_mask = None
            #attn_mask = None
        else: # LDA
            I, Gh, Gw = self.interval, group_size, group_size
            Rh, Rw = Hp // (Gh * I), Wp // (Gw * I)
            x = x.reshape(B, Rh, Gh, I, Rw, Gw, I, C).permute(0, 1, 4, 3, 6, 2, 5, 7).contiguous()
            x = x.reshape(B * Rh * Rw * I * I, Gh * Gw, C)
            nG = I ** 2 * Rh * Rw
            # attn_mask
            if pad_r > 0 or pad_b > 0:
                mask = mask.reshape(1, Rh, Gh, I, Rw, Gw, I, 1).permute(0, 1, 4, 3, 6, 2, 5, 7).contiguous()
                mask = mask.reshape(nG, 1, Gh * Gw)
                attn_mask = torch.zeros((nG, Gh * Gw, Gh * Gw), device=x.device)
                attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
            else:
                attn_mask = None
        attn_mask = None
        # multi-head self-attention
        x = self.attn(x, Gh, Gw, mask=attn_mask)  # nG*B, G*G, C
        
        # ungroup embeddings
        if self.lsda_flag == 0:
            x = x.reshape(B, Hp // G, Wp // G, G, G, C).permute(0, 1, 3, 2, 4, 5).contiguous() # B, Hp//G, G, Wp//G, G, C
        else:
            x = x.reshape(B, Rh, Rw, I, I, Gh, Gw, C).permute(0, 1, 5, 3, 2, 6, 4, 7).contiguous() # B, Rh, Gh, I, Rw, Gw, I, C
        x = x.reshape(B, Hp, Wp, C)

        # remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        #
        # # cooling layer
        # if self.use_extra_conv:
        #     x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        #     x = self.ex_conv(x)
        #     x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        #     x = self.ex_ln(x)

        return x


    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"group_size={self.group_size}, lsda_flag={self.lsda_flag}, mlp_ratio={self.mlp_ratio}, " \
               f"interval={self.interval}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # Attention
        size_div = self.interval * self.group_size if self.lsda_flag == 1 else self.group_size
        Hp = math.ceil(H / size_div) * size_div
        Wp = math.ceil(W / size_div) * size_div
        Gh = Gw = self.group_size
        nG = Hp * Wp / Gh / Gw
        attn_flops, attn_excluded_flops = self.attn.flops(Gh * Gw)
        flops += nG * attn_flops
        excluded_flops = nG * attn_excluded_flops
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops, excluded_flops


class CrossFormerBlock1(nn.Module):

    def __init__(self, dim=64, input_resolution=[40, 40], num_heads=2, group_size=20, interval=8, lsda_flag=0,
                 mlp_ratio=1., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_patch_size=1,
                 pad_type=0, use_extra_conv=True, use_cpe=False, no_mask=False, adaptive_interval=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.group_size = group_size
        self.interval = interval
        self.lsda_flag = lsda_flag
        self.mlp_ratio = mlp_ratio
        self.num_patch_size = num_patch_size
        self.pad_type = pad_type
        self.use_extra_conv = use_extra_conv
        self.use_cpe = use_cpe
        self.adaptive_interval = adaptive_interval

        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            position_bias=(not use_cpe))


    def forward(self, x, H, W):

        B, L, C = x.shape


        if min(H, W) <= self.group_size:

            self.lsda_flag = 0
            group_size = max(H, W)
        else:
            group_size = self.group_size


        x = x.view(B, H, W, C)



        # padding
        size_div = self.interval * group_size if self.lsda_flag == 1 else group_size
        pad_l = pad_t = 0
        pad_r = (size_div - W % size_div) % size_div
        pad_b = (size_div - H % size_div) % size_div
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape


        G = Gh = Gw = 2
        x = x.reshape(B, Hp // G, G, Wp // G, G, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(B * Hp * Wp // G ** 2, G ** 2, C)
        attn_mask = None
        # multi-head self-attention
        x = self.attn(x, Gh, Gw, mask=attn_mask)  # nG*B, G*G, C
        x = x.reshape(B, Hp // G, Wp // G, G, G, C).permute(0, 1, 3, 2, 4,5).contiguous()  # B, Hp//G, G, Wp//G, G, C
        x = x.reshape(B, Hp, Wp, C)

        # remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
##############################################
        I2, Gh2, Gw2 = self.interval, 2, 2
        Rh, Rw = Hp // (Gh2 * I2), Wp // (Gw2 * I2)
        x = x.reshape(B, Rh, Gh2, I2, Rw, Gw2, I2, C).permute(0, 1, 4, 3, 6, 2, 5, 7).contiguous()
        x = x.reshape(B * Rh * Rw * I2 * I2, Gh2 * Gw2, C)
        x = self.attn(x, Gh2, Gw2, mask=attn_mask)  # nG*B, G*G, C
        x = x.reshape(B, Rh, Rw, I2, I2, Gh2, Gw2, C).permute(0, 1, 5, 3, 2, 6, 4,7).contiguous()
        x = x.reshape(B, Hp, Wp, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        return x














# if __name__ == '__main__':
#     fea = torch.randn(1, 64, 160, 160).view(1,25600,64)
#
#     net = CrossFormerBlock()
#
#     #dummy_x = torch.randn(1, 256, 160, 160)
#
#     x_1 = net(fea,160 // 1, 160 // 1)
#
#     print(net.flops())
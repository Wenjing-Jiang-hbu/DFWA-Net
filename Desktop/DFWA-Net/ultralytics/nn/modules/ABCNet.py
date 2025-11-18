import torch
import torch.nn as nn
from .ABCModule import conv_block, up_conv, _upsample_like, conv_relu_bn, dconv_block
from einops import rearrange
from .gcn.layers.GConv2 import GConv2

from pytorch_wavelets import DWTForward, DWTInverse

import torch.nn.functional as F

from .crossformer_backbone_det import CrossFormerBlock,CrossFormerBlock1




class CWST2(nn.Module):
    def __init__(self, in_dim, out_dim, size):
        super(CWST2, self).__init__()
        self.WTF = DWTForward(J=1, wave="haar")
        self.WTI = DWTInverse(mode='zero', wave="haar")
        self.wdim = [in_dim, in_dim]

        len = int(size / 2)
        self.trans1 = CrossFormerBlock1(dim=int(in_dim/4), input_resolution=[len, len], num_heads=1, group_size=2, interval=len,
                                       lsda_flag=1)
        self.convs = GConv2(int((in_dim/1) / 4), int((in_dim/1) / 4), 5, M=4, padding=2)

        self.fusion = ChannelSelect(in_dim)

    def forward(self, x):

        xl1, xh1 = self.WTF(x)

        b, c, n, h, w = xh1[0].shape
        channels_per_group = xh1[0].shape[2] // 3
        s1, s2, s3 = torch.split(xh1[0], channels_per_group, 2)
        s1 = s1.view(b, -1, h, w)
        s2 = s2.view(b, -1, h, w)
        s3 = s3.view(b, -1, h, w)

        top = torch.cat((xl1, s1), dim=3)  # dim=3 是宽度的拼接
        bottom = torch.cat((s2, s3), dim=3)  # dim=3 是宽度的拼接

        # 最后在批次维度上拼接 top 和 bottom
        feature_hl = torch.cat((top, bottom), dim=2)  # dim=2 是高度的拼接
        feature_hl = feature_hl.view(b, -1, c)

        channels_per_group1 = feature_hl.shape[2] // 4
        feature_hl_1, feature_hl_2, feature_hl_3, feature_hl_4= torch.split(feature_hl, channels_per_group1, 2)

        trans_fea11 = self.trans1(feature_hl_1,2*h,2*w).view(b,int(c//4),2*h,2*w)
        trans_fea12 = self.trans1(feature_hl_2,2*h,2*w).view(b,int(c//4),2*h,2*w)
        trans_fea13 = self.trans1(feature_hl_3,2*h,2*w).view(b,int(c//4),2*h,2*w)
        trans_fea14 = self.trans1(feature_hl_4,2*h,2*w).view(b,int(c//4),2*h,2*w)


        trans_fea2 = torch.cat((trans_fea11, trans_fea12, trans_fea13,trans_fea14), dim=1)

        yl1 = trans_fea2[:, :, :h, :w] + xl1  # 左上角+
        x2_new = trans_fea2[:, :, :h, w:] + s1  # 右上角
        x3_new = trans_fea2[:, :, h:, :w] + s2  # 左下角
        x4_new = trans_fea2[:, :, h:, w:] + s3  # 右下角


        yh1 = torch.stack(
            [x2_new, x3_new, x4_new], dim=2
        )
        # # #
        yh1_ = [None]
        yh1_[0] = yh1.view(b, -1, 3, h, w)

        transy = self.WTI((yl1, yh1_))
        convsy = self.convs(x)

        out = self.fusion(transy, convsy)
        return out




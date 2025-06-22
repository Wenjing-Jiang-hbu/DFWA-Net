# @Time    : 2023/3/17 15:56
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : ABCNet.py
# @Software: PyCharm
import torch
import torch.nn as nn
from .ABCModule import conv_block, up_conv, _upsample_like, conv_relu_bn, dconv_block
from einops import rearrange
from .gcn.layers.GConv2 import GConv2

from pytorch_wavelets import DWTForward, DWTInverse

from.SFS_Conv import SFS_Conv, GaborFPU
import torch.nn.functional as F

from .crossformer_backbone_det import CrossFormerBlock,CrossFormerBlock1










class CWST1(nn.Module):
    def __init__(self, in_dim, out_dim, size):
        super(CWST1, self).__init__()
        self.WTF = DWTForward(J=1, wave="haar")
        self.WTI = DWTInverse(mode='zero', wave="haar")
        self.wdim = [in_dim, in_dim]

        len = int(size / 2)
        self.trans1 = CrossFormerBlock(dim=int(in_dim//4), input_resolution=[len,len], num_heads=2, group_size=5, interval=8, lsda_flag=0)
        self.trans2 = CrossFormerBlock(dim=int(in_dim//4), input_resolution=[len,len], num_heads=2, group_size=2, interval=len, lsda_flag=1)
        self.convs = GConv2(int(in_dim / 2), int(in_dim / 2), 3, M=2, padding=1)
        self.fusion = ChannelSelect(in_dim)

    def forward(self, x):

        xl1, xh1 = self.WTF(x)
        # yl1 = self.trans1(xl1)

        b,c,n,h,w = xh1[0].shape
        channels_per_group = xh1[0].shape[2] // 3
        s1, s2, s3 = torch.split(xh1[0], channels_per_group, 2)
        s1 = s1.view(b,-1,h,w)
        s2 = s2.view(b,-1,h,w)
        s3 = s3.view(b,-1,h,w)

        top = torch.cat((xl1, s1), dim=3)  # dim=3 是宽度的拼接

        bottom = torch.cat((s2, s3), dim=3)  # dim=3 是宽度的拼接

        # 最后在批次维度上拼接 top 和 bottom
        feature_hl = torch.cat((top, bottom), dim=2)  # dim=2 是高度的拼接
        feature_hl = feature_hl.view(b,-1,c)

        channels_per_group1 = feature_hl.shape[2] // 4
        feature_hl_1, feature_hl_2, feature_hl_3, feature_hl_4= torch.split(feature_hl, channels_per_group1, 2)



        trans_fea1_1 = self.trans1(feature_hl_1,2*h,2*w)
        trans_fea1_2 = self.trans1(feature_hl_2,2*h,2*w)
        trans_fea1_3 = self.trans1(feature_hl_3,2*h,2*w)
        trans_fea1_4 = self.trans1(feature_hl_4,2*h,2*w)


        trans_fea11 = self.trans2(trans_fea1_1,2*h,2*w).view(b,int(c//4),2*h,2*w)
        trans_fea12 = self.trans2(trans_fea1_2,2*h,2*w).view(b,int(c//4),2*h,2*w)
        trans_fea13 = self.trans2(trans_fea1_3,2*h,2*w).view(b,int(c//4),2*h,2*w)
        trans_fea14 = self.trans2(trans_fea1_4,2*h,2*w).view(b,int(c//4),2*h,2*w)



        trans_fea2 = torch.cat((trans_fea11, trans_fea12, trans_fea13,trans_fea14), dim=1)

        yl1 = trans_fea2[:, :, :h, :w]  # 左上角+
        x2_new = trans_fea2[:, :, :h, w:]  # 右上角
        x3_new = trans_fea2[:, :, h:, :w]  # 左下角
        x4_new = trans_fea2[:, :, h:, w:]# 右下角

        # s = torch.mean(xh1[0],2)
        # sy = self.trans2(s).view(b,-1,1,h,w)
        #
        # sy1 = s1+sy
        # sy2 = s2+sy
        # sy3 = s3+sy
        # # sy2 = self.trans2(s2.view(b,-1,h,w))
        # # sy3 = self.trans2(s3.view(b,-1,h,w))
        yh1 = torch.stack(
            [x2_new, x3_new, x4_new], dim=2
        )
        # # #
        yh1_ = [None]
        yh1_[0] = yh1.view(b,-1,3,h,w)




        transy = self.WTI((yl1, xh1))
        convsy = self.convs(x)
        out = self.fusion(transy, convsy)
        return out



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

        transy = self.WTI((yl1, xh1))
        convsy = self.convs(x)

        out = self.fusion(transy, convsy)
        return out




# class CWST2(nn.Module):#XIAORONG2_1
#     def __init__(self, in_dim, out_dim, size):
#         super(CWST2, self).__init__()
#         self.convs = GConv2(int((in_dim/1) / 4), int((in_dim/1) / 4), 5, M=4, padding=2)
#
#     def forward(self, x):
#         convsy = self.convs(x)
#
#         out = convsy
#         return out





class CWST3(nn.Module):
    def __init__(self, in_dim, out_dim, size):
        super(CWST3, self).__init__()
        self.WTF = DWTForward(J=1, wave="haar")
        self.WTI = DWTInverse(mode='zero', wave="haar")
        self.wdim = [in_dim, in_dim]

        len = int(size / 2)
        self.trans1 = CrossFormerBlock(dim=int(in_dim // 2), input_resolution=[len, len], num_heads=2, group_size=5,
                                       interval=8, lsda_flag=0)
        self.trans2 = CrossFormerBlock(dim=int(in_dim // 2), input_resolution=[len, len], num_heads=2, group_size=2,
                                       interval=len, lsda_flag=1)
        self.convs = GConv2(int(in_dim / 4), int(in_dim / 4), 3, M=4, padding=1)
        self.fusion = ChannelSelect(in_dim)

    def forward(self, x):
        xl1, xh1 = self.WTF(x)
        # yl1 = self.trans1(xl1)

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

        channels_per_group1 = feature_hl.shape[2] // 2
        feature_hl_1, feature_hl_2= torch.split(
            feature_hl, channels_per_group1, 2)

        trans_fea1_1 = self.trans1(feature_hl_1, 2 * h, 2 * w)
        trans_fea1_2 = self.trans1(feature_hl_2, 2 * h, 2 * w)



        trans_fea11 = self.trans2(trans_fea1_1, 2 * h, 2 * w).view(b, int(c // 2), 2 * h, 2 * w)
        trans_fea12 = self.trans2(trans_fea1_2, 2 * h, 2 * w).view(b, int(c // 2), 2 * h, 2 * w)



        trans_fea2 = torch.cat(
            (trans_fea11, trans_fea12),
            dim=1)

        yl1 = trans_fea2[:, :, :h, :w]  # 左上角+
        x2_new = trans_fea2[:, :, :h, w:]  # 右上角
        x3_new = trans_fea2[:, :, h:, :w]  # 左下角
        x4_new = trans_fea2[:, :, h:, w:]  # 右下角

        # s = torch.mean(xh1[0],2)
        # sy = self.trans2(s).view(b,-1,1,h,w)
        #
        # sy1 = s1+sy
        # sy2 = s2+sy
        # sy3 = s3+sy
        # # sy2 = self.trans2(s2.view(b,-1,h,w))
        # # sy3 = self.trans2(s3.view(b,-1,h,w))
        yh1 = torch.stack(
            [x2_new, x3_new, x4_new], dim=2
        )
        # # #
        yh1_ = [None]
        yh1_[0] = yh1.view(b, -1, 3, h, w)

        transy = self.WTI((yl1, xh1))
        convsy = self.convs(x)
        out = self.fusion(transy, convsy)
        return out


class CWST4(nn.Module):
    def __init__(self, in_dim, out_dim, size):
        super(CWST4, self).__init__()
        self.WTF = DWTForward(J=1, wave="coif2")
        self.WTI = DWTInverse(mode='zero', wave="coif2")
        self.wdim = [in_dim, in_dim]

        len = int(size / 2) + 5
        self.trans1 = ConvAttention(self.wdim[0], int(len * len), int(len))
        self.convs = GConv2(int(in_dim / 4), int(in_dim / 4), 5, M=4, padding=2)
        self.fusion = ChannelSelect(in_dim)

    def forward(self, x):
        xl1, xh1 = self.WTF(x)
        yl1 = self.trans1(xl1)
        transy = self.WTI((yl1, xh1))
        convsy = self.convs(x)
        out = self.fusion(transy, convsy)
        return out




class Attention(nn.Module):
    def __init__(self, in_dim, in_feature, out_feature):
        super(Attention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
        self.query_line = nn.Linear(in_features=in_feature, out_features=out_feature)
        self.key_line = nn.Linear(in_features=in_feature, out_features=out_feature)
        self.s_conv = nn.Conv2d(in_channels=1, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = rearrange(self.query_line(rearrange(self.query_conv(x), 'b 1 h w -> b (h w)')), 'b h -> b h 1')
        k = rearrange(self.key_line(rearrange(self.key_conv(x), 'b 1 h w -> b (h w)')), 'b h -> b 1 h')
        att = rearrange(torch.matmul(q, k), 'b h w -> b 1 h w')
        att = self.softmax(self.s_conv(att))
        return att


class DEPTHWISECONV(nn.Module):
    def __init__(self,in_ch,out_ch,kernel):
        super(DEPTHWISECONV, self).__init__()


        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=kernel,
                                    stride=1,
                                    padding="same",
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class Conv2(nn.Module):
    def __init__(self, in_dim):
        super(Conv2, self).__init__()
        # self.convs = nn.ModuleList([conv_relu_bn(in_dim, in_dim, 1) for _ in range(3)])
        self.convs = DEPTHWISECONV(in_dim,in_dim,3,padding=1)

    def forward(self, x):
        # for conv in self.convs:
        #     x = conv(x)
        x = self.convs(x)
        return x

class Conv(nn.Module):
    def __init__(self, in_dim):
        super(Conv, self).__init__()
        #self.convs = nn.ModuleList([conv_relu_bn(in_dim, in_dim, 1) for _ in range(3)])
        #self.convs =  DEPTHWISECONV(in_dim,in_dim,3,padding=1)
        #self.convs = GaborFPU(int(in_dim),int(in_dim))
        #self.convs = nn.Conv2d(in_dim,in_dim,1,1)
        self.convs = GConv2(int(in_dim/4), int(in_dim/4), 3,M=4, padding=1)

    def forward(self, x):
        # for conv in self.convs:
        #     x = conv(x)
        x = self.convs(x)
        # x = self.convs(x)
        return x



class DConv(nn.Module):
    def __init__(self, in_dim):
        super(DConv, self).__init__()
        #dilation = [2, 4, 2]
        self.dconvs = GConv2(int(in_dim/8), int(in_dim/8), 3,M=8, padding=1)
        #self.dconvs = GaborFPU(int(in_dim), int(in_dim))
        #self.dconvs = nn.Conv2d(in_dim,in_dim,1,1)

    def forward(self, x):
        # for dconv in self.dconvs:
        #     x = dconv(x)
        x = self.dconvs(x)
        # x = self.dconvs(x)
        return x


# class ConvAttention(nn.Module):
    # def __init__(self, in_dim, in_feature, out_feature):
    #     super(ConvAttention, self).__init__()
    #     # self.conv = Conv(in_dim)
    #     # self.dconv = DConv(in_dim)
    #     self.att = Attention(in_dim, in_feature, out_feature)
    #
    #
    # def forward(self, q,k,v):
    #     # q = self.conv(x)
    #     # k = self.dconv(x)
    #     # v = q + k
    #     out = [None]
    #     att = self.att(q,k)
    #     h1 = torch.matmul(att, q)+q
    #     h2 = torch.matmul(att, k)+k
    #     h3 = torch.matmul(att, v)+v
    #     out[0] = torch.stack((h1,h2,h3),2)
    #     return out

class ConvAttention(nn.Module):
    def __init__(self, in_dim, in_feature, out_feature):
        super(ConvAttention, self).__init__()
        self.conv = Conv(in_dim)
        self.dconv = DConv(in_dim)
        self.att = Attention(in_dim, in_feature, out_feature)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        q = self.conv(x)
        k = self.dconv(x)
        v = 0.2*q + 0.6*k
        att = self.att(x)
        out = torch.matmul(att, v)
        return self.gamma*out+k


class myConvAttention(nn.Module):
    def __init__(self, in_dim, in_feature, out_feature):
        super(myConvAttention, self).__init__()
        self.convq = DEPTHWISECONV(in_dim,in_dim)
        self.convk = DEPTHWISECONV(in_dim,in_dim)
        self.convv = DEPTHWISECONV(in_dim, in_dim)
        # self.convq = nn.Conv2d(in_dim,in_dim,3,1,1)
        # self.convk = nn.Conv2d(in_dim,in_dim,3,1,1)
        # self.convv = nn.Conv2d(in_dim,in_dim,3,1,1)
        # self.dconv = DConv(in_dim)
        self.att = Attention(in_dim, in_feature, out_feature)
        # self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        # x = torch.randn(1, 64, 160, 160)
        q = self.convq(x)
        k = self.convk(x)
        v = q + k
        att = self.att(q, k)
        out = torch.matmul(att, v)
        return out




class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForward, self).__init__()
        self.conv = conv_relu_bn(in_dim, out_dim, 1)
        # self.x_conv = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.x_conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        x = self.x_conv(x)
        return x + out



class ConvTransformer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ConvTransformer, self).__init__()

        # self.feedforward = FeedForward(in_dim, out_dim)

        # self.dim_cwt = int(in_dim // 1)
        # self.dim_untouched = in_dim - self.dim_cwt
        self.attention = DEPTHWISECONV(in_dim,out_dim,3)
    def forward(self, x):
        #xa, xna = torch.split(x, [self.dim_cwt, self.dim_untouched], dim=1)

        x = self.attention(x)
        # out = self.feedforward(x)
        #out = torch.cat((xa, xna), 1)

        return x

# def norm_tensor(x):
#     min_value = x.min()
#     max_value = x.max()
#     out = (x - min_value) / (max_value - min_value)
#     return out

class DHL(nn.Module):
    def __init__(self, in_dim, wt_type,feature):
        super(DHL, self).__init__()

        self.WTF = DWTForward(J=1, wave=wt_type)
        self.WTI = DWTInverse(mode='zero', wave=wt_type)
        self.conv = DEPTHWISECONV(in_dim,in_dim,kernel=3)
        self.attn = ConvAttention(in_dim,int(feature*feature/4),int(feature/2))
    def forward(self, x):
        xl, xh = self.WTF(x)
        yl = self.conv(xl)
        q = xh[0][:, :, 0, :, :]
        k = xh[0][:, :, 1, :, :]
        v = xh[0][:, :, 2, :, :]
        yh = self.attn(q,k,v)

        output = self.WTI((yl, yh))

        return output



class CWT1(nn.Module):
    def __init__(self, in_dim, out_dim, feature):
        super(CWT1, self).__init__()


        self.dim_cwt = int(in_dim/4)
        self.dim_Ncwt = in_dim-int(in_dim/4)
        self.cwt = DHL(self.dim_cwt, wt_type='haar',feature=feature)
        self.Ncwt = GConv2(int(self.dim_Ncwt/4),int(self.dim_Ncwt/4),3,padding=1)
        self.conv = DEPTHWISECONV(in_dim,out_dim,1)
    def forward(self, x):

        x1, x2 = torch.split(x, [self.dim_cwt,self.dim_Ncwt], dim=1)

        x1 = self.cwt(x1)
        x2 = self.Ncwt(x2)

        out_ = torch.concat((x1, x2), dim=1)
        out = self.conv(out_)
        return x


class CWT2(nn.Module):
    def __init__(self, in_dim, out_dim,size):
        super(CWT2, self).__init__()
        feature = int(size/2)
        # pre = torch.randn(1,1,size,size)
        self.WTF = DWTForward(J=1, wave="haar")
        self.WTI = DWTInverse(mode='zero', wave="haar")
        # xl,xh = self.WTF(pre)
        self.wdim = [in_dim,int(in_dim),int(in_dim)]
        self.lconv1 = ConvAttention(self.wdim[0],feature*feature,feature)
        self.hconv1 = GaborFPU(self.wdim[0],self.wdim[0])
        self.lconv2 = ConvAttention(self.wdim[1],int(feature*feature/4),int(feature/2))
        self.hconv2 = GaborFPU(self.wdim[1], self.wdim[1])
        self.lconv3 = DEPTHWISECONV(self.wdim[2],self.wdim[2],5)
        self.hconv3 = GaborFPU(self.wdim[2],self.wdim[2])
        self.conv = nn.Conv2d(3*self.wdim[2], out_dim, 1)
        # self.hconv1 = GConv2(int(in_dim/4),int(out_dim/4),5,nScale=5,padding=2)
        # self.hconv2 = GConv2(int(in_dim/4),int(out_dim/4),7,nScale=7,padding=3)
        # self.hconv3 = GConv2(int(in_dim/4),int(out_dim/4),9,nScale=9,padding=4)

    def forward(self, x):

        xl1, xh1 = self.WTF(x)
        yl1 = self.lconv1(xl1)
        yh1 = self.hconv1(xh1[0])

        xl2, xh2 = self.WTF(yl1)
        yl2 = self.lconv2(xl2)
        yh2 = self.hconv2(xh2[0])

        xl3, xh3 = self.WTF(yl2)
        yl3 = self.lconv3(xl3)
        yh3 = self.hconv3(xh3[0])

        yh3_ = [None]
        yh3_[0] = yh3
        yh2_ = [None]
        yh2_[0] = yh2
        yh1_ = [None]
        yh1_[0] = yh1
        y3 = F.interpolate(self.WTI((yl3,yh3_)),scale_factor=4)
        y2 = F.interpolate(self.WTI((yl2,yh2_)),scale_factor=2)
        y1 = self.WTI((yl1,yh1_))



        # yh1 = self.hconv1(torch.squeeze(torch.mean((xh[0]), dim=2), 2))
        # yh2 = self.hconv2(torch.squeeze(torch.mean((xh[1]), dim=2), 2))
        # yh3 = self.hconv3(torch.squeeze(torch.mean((xh[2]), dim=2), 2))
        # # yh4 = self.hconv4(torch.squeeze(torch.mean((xh[3]), dim=2), 2))
        # yh = [xh[0] * torch.unsqueeze(yh1, 2), xh[1] * torch.unsqueeze(yh2, 2), xh[2] * torch.unsqueeze(yh3, 2)]

        y = self.conv(torch.cat([y1, y2, y3], dim=1))

        return y


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class myConv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv0 = nn.Conv2d(c1,c2,1,1)
        self.conv = DEPTHWISECONV(c2,c2,k)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(self.conv0(x))))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))





class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = myConv(c1, c_, k[0], 1)
        self.cv2 = myConv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class myC2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

#
# class ChannelSelect2(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelSelect2, self).__init__()
#
#         # self.advavg = nn.AdaptiveAvgPool2d(1)
#         # self.conv = DEPTHWISECONV(in_planes*2,in_planes*2,1)
#
#         self.conv1 = nn.Conv2d(in_planes, in_planes, 1)
#         self.conv2 = nn.Conv2d(in_planes, in_planes, 1)
#         self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
#     def forward(self, x, y):
#
#         attn1 = self.conv1(x)
#         attn2 = self.conv2(y)
#         attn = torch.cat([attn1, attn2], dim=1)
#         avg_attn = torch.mean(attn, dim=1, keepdim=True)
#         max_attn, _ = torch.max(attn, dim=1, keepdim=True)
#         agg = torch.cat([avg_attn, max_attn], dim=1)
#         sig = self.conv_squeeze(agg).sigmoid()
#         output = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
#         # attn = self.conv(attn)
#         # return x * attn
#
#         return output



class ChannelSelect(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelSelect, self).__init__()

        self.advavg = nn.AdaptiveAvgPool2d(1)
        self.conv = DEPTHWISECONV(in_planes*2,in_planes*2,1)
    def forward(self, x, y):
        # b,c,h,w = x.shape
        # xfc = x.view(b,c,-1)
        # yfc = y.view(b,c,-1)
        xy = torch.cat([x, y], dim=1)
        ca = F.softmax(self.conv(self.advavg(xy)), dim=1).sigmoid()
        cax,cay = torch.split(ca, ca.size(1) // 2, dim=1)

        mask1 = cax > cay  # 如果 score1 大于 score2，选择 x1
        mask2 = cax < cay  # 如果 score1 小于 score2，选择 x2
        mask_equal = cax == cay  # 如果 score1 等于 score2，选择 x1 + x2
        # 通过掩膜选择特征图
        output = torch.zeros_like(x)  # 初始化输出
        output += mask1 * x  # score1 > score2 时选择 x1
        output += mask2 * y  # score1 < score2 时选择 x2
        output += mask_equal * ((x + y)/2)  # score1 == score2 时选择 x1 + x2

        return output



class ChannelSelect2(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelSelect2, self).__init__()

        self.advavg = nn.AdaptiveAvgPool2d(1)
        self.conv = DEPTHWISECONV(in_planes*2,in_planes*2,1)
    def forward(self, x, y):
        # b,c,h,w = x.shape
        # xfc = x.view(b,c,-1)
        # yfc = y.view(b,c,-1)
        xy = torch.cat([x, y], dim=1)
        ca = F.softmax(self.conv(self.advavg(xy)), dim=1).sigmoid()
        cax,cay = torch.split(ca, ca.size(1) // 2, dim=1)

        output = x*cax+y*cay

        return output



class CWT3(nn.Module):
    def __init__(self, in_dim, out_dim, size):
        super(CWT3, self).__init__()

        # pre = torch.randn(1,1,size,size)

        # xl,xh = self.WTF(pre)
        self.wdim = [in_dim, in_dim]
        #self.lconv1 = DEPTHWISECONV(self.wdim[0], 2*self.wdim[0], 5)
        # self.lconv1 = ConvAttention(self.wdim[0], int(size * size / 4), int(size / 2))
        # self.lconv1 = GaborFPU(self.wdim[0], self.wdim[0])
        # self.hconv1 = ConvAttention(self.wdim[0]*3, int(size * size / 4), int(size / 2))
        #self.lconv = ConvAttention(int(self.wdim[0]/1), int(size * size / 4), int(size / 2))
        # self.hconv3 = ConvAttention(self.wdim[0], int(size * size / 4), int(size / 2))
        #self.conv = nn.Conv2d(int(self.wdim[0]/1), int(self.wdim[0]/1),1,1,0)
        # self.lconv2 = DEPTHWISECONV(self.wdim[1], self.wdim[1], 5)
        # self.hconv2 = GaborFPU(self.wdim[1], self.wdim[1])
        # self.lconv3 = DEPTHWISECONV(self.wdim[2],self.wdim[2],5)
        # self.hconv3 = GaborFPU(self.wdim[2],self.wdim[2])
        # self.hconv1 = GConv2(int(in_dim/4),int(out_dim/4),3,padding=1)
        # self.hconv2 = GConv2(int(in_dim/4),int(out_dim/4),5,padding=2)
        # self.hconv3 = GConv2(int(in_dim/4),int(out_dim/4),7,padding=3)
        len=int(size/1)
        self.trans1 = ConvAttention(self.wdim[0], int(len * len), int(len))
        #self.trans2 = ConvAttention(self.wdim[0], int(size * size/4), int(size/2))
        self.convs = GConv2(int(in_dim/4), int(in_dim/4), 5,M=4, padding=2)
        # self.conv_squeeze = DEPTHWISECONV(2, 2, 1)
        self.fusion = ChannelSelect(in_dim)
        #self.fusion = nn.Conv2d(2*in_dim,in_dim,1)
    def forward(self, x):

        # b,c,n,h,w = xh1[0].shape
        # channels_per_group = xh1[0].shape[2] // 3
        # s1, s2, s3 = torch.split(xh1[0], channels_per_group, 2)
        # s = torch.mean(xh1[0],2)
        # sy = self.trans2(s).view(b,-1,1,h,w)
        #
        # sy1 = s1+sy
        # sy2 = s2+sy
        # sy3 = s3+sy
        # # sy2 = self.trans2(s2.view(b,-1,h,w))
        # # sy3 = self.trans2(s3.view(b,-1,h,w))
        # yh1 = torch.stack(
        #     [sy1, sy2, sy3], dim=2
        # )
        # # #
        # yh1_ = [None]
        # yh1_[0] = yh1.view(b,-1,3,h,w)

        transy = self.trans1(x)

        # transy = self.trans(x)




        convsy = self.convs(x)

        # channels_per_group = x.shape[1] // 4
        # x1 = x[:,0:channels_per_group,:,:]
        # x2 = x[:,channels_per_group:2*channels_per_group,:,:]
        # x3 = x[:,2*channels_per_group:3*channels_per_group,:,:]
        # x4 = x[:,3*channels_per_group:,:,:]
        # y1 = self.hconv2(x1)
        # y2 = self.hconv2(x2)
        # y3 = self.hconv2(x3)
        # y4 = self.hconv2(x4)
        #
        # xl1, xh1 = self.WTF(x)
        # #yl1 = self.lconv(xl1)
        #
        #
        # b,c,n,h,w = xh1[0].shape
        # channels_per_group = xh1[0].shape[2] // 3
        # s1, s2, s3 = torch.split(xh1[0], channels_per_group, 2)
        # sy1 = self.hconv1(s1.view(b,-1,h,w))
        # sy2 = self.hconv2(s2.view(b,-1,h,w))
        # sy3 = self.hconv3(s3.view(b,-1,h,w))

        # channels_per_group = xh1[0].shape[2] // 3
        # s1, s2, s3 = torch.split(xh1[0], channels_per_group, 2)
        #
        # b,c,n,h,w = xh1[0].shape
        # sy1 = self.hconv2(s1.view(b,-1,h,w))
        # sy2 = self.hconv2(s2.view(b,-1,h,w))
        # sy3 = self.hconv2(s3.view(b,-1,h,w))
        #
        #
        # yh1 = torch.stack(
        #     [sy1, sy2, sy3], dim=2
        # )
        # # #
        # yh1_ = [None]
        # yh1_[0] = yh1
        #
        # y = self.WTI((xl1, yh1_))
        #y = self.conv(y1)
        # out = F.interpolate(yl1,scale_factor=2)


        # yh1 = self.hconv1(torch.squeeze(torch.mean((xh[0]), dim=2), 2))
        # yh2 = self.hconv2(torch.squeeze(torch.mean((xh[1]), dim=2), 2))
        # yh3 = self.hconv3(torch.squeeze(torch.mean((xh[2]), dim=2), 2))
        # # yh4 = self.hconv4(torch.squeeze(torch.mean((xh[3]), dim=2), 2))
        # yh = [xh[0] * torch.unsqueeze(yh1, 2), xh[1] * torch.unsqueeze(yh2, 2), xh[2] * torch.unsqueeze(yh3, 2)]

        #

        #out = torch.concat((transy,convsy),1)
        out = self.fusion(transy,convsy)


        # avg_attn = torch.mean(out, dim=1, keepdim=True)
        # max_attn, _ = torch.max(out, dim=1, keepdim=True)
        # agg = torch.cat([avg_attn, max_attn], dim=1)
        # sig = self.conv_squeeze(agg).sigmoid()
        #
        # attn = transy * sig[:, 0, :, :].unsqueeze(1) + convsy * sig[:, 1, :, :].unsqueeze(1)
        # attn = self.conv(attn)
        # return x * attn

        #
        return out






class CWT4(nn.Module):
    def __init__(self, in_dim, out_dim, size):
        super(CWT4, self).__init__()

        # pre = torch.randn(1,1,size,size)
        self.WTF = DWTForward(J=1, wave="haar")
        self.WTI = DWTInverse(mode='zero', wave="haar")
        # xl,xh = self.WTF(pre)
        self.wdim = [in_dim, in_dim]
        #self.lconv1 = DEPTHWISECONV(self.wdim[0], 2*self.wdim[0], 5)
        # self.lconv1 = ConvAttention(self.wdim[0], int(size * size / 4), int(size / 2))
        # self.lconv1 = GaborFPU(self.wdim[0], self.wdim[0])
        # self.hconv1 = ConvAttention(self.wdim[0]*3, int(size * size / 4), int(size / 2))
        #self.lconv = ConvAttention(int(self.wdim[0]/1), int(size * size / 4), int(size / 2))
        # self.hconv3 = ConvAttention(self.wdim[0], int(size * size / 4), int(size / 2))
        #self.conv = nn.Conv2d(int(self.wdim[0]/1), int(self.wdim[0]/1),1,1,0)
        # self.lconv2 = DEPTHWISECONV(self.wdim[1], self.wdim[1], 5)
        # self.hconv2 = GaborFPU(self.wdim[1], self.wdim[1])
        # self.lconv3 = DEPTHWISECONV(self.wdim[2],self.wdim[2],5)
        # self.hconv3 = GaborFPU(self.wdim[2],self.wdim[2])
        # self.hconv1 = GConv2(int(in_dim/4),int(out_dim/4),3,padding=1)
        # self.hconv2 = GConv2(int(in_dim/4),int(out_dim/4),5,padding=2)
        # self.hconv3 = GConv2(int(in_dim/4),int(out_dim/4),7,padding=3)
        len=int(size/2)
        self.trans1 = ConvAttention(self.wdim[0], int(len * len), int(len))
        #self.trans2 = ConvAttention(self.wdim[0], int(size * size/4), int(size/2))
        self.convs = GConv2(int(in_dim/4), int(in_dim/4), 5,M=4, padding=2)
        # self.conv_squeeze = DEPTHWISECONV(2, 2, 1)
        self.fusion = ChannelSelect(in_dim)
        #self.fusion = nn.Conv2d(2*in_dim,in_dim,1)
    def forward(self, x):
        xl1, xh1 = self.WTF(x)
        yl1 = self.trans1(xl1)
        # b,c,n,h,w = xh1[0].shape
        # channels_per_group = xh1[0].shape[2] // 3
        # s1, s2, s3 = torch.split(xh1[0], channels_per_group, 2)
        # s = torch.mean(xh1[0],2)
        # sy = self.trans2(s).view(b,-1,1,h,w)
        #
        # sy1 = s1+sy
        # sy2 = s2+sy
        # sy3 = s3+sy
        # # sy2 = self.trans2(s2.view(b,-1,h,w))
        # # sy3 = self.trans2(s3.view(b,-1,h,w))
        # yh1 = torch.stack(
        #     [sy1, sy2, sy3], dim=2
        # )
        # # #
        # yh1_ = [None]
        # yh1_[0] = yh1.view(b,-1,3,h,w)

        transy = self.WTI((yl1, xh1))

        # transy = self.trans(x)




        convsy = self.convs(x)

        # channels_per_group = x.shape[1] // 4
        # x1 = x[:,0:channels_per_group,:,:]
        # x2 = x[:,channels_per_group:2*channels_per_group,:,:]
        # x3 = x[:,2*channels_per_group:3*channels_per_group,:,:]
        # x4 = x[:,3*channels_per_group:,:,:]
        # y1 = self.hconv2(x1)
        # y2 = self.hconv2(x2)
        # y3 = self.hconv2(x3)
        # y4 = self.hconv2(x4)
        #
        # xl1, xh1 = self.WTF(x)
        # #yl1 = self.lconv(xl1)
        #
        #
        # b,c,n,h,w = xh1[0].shape
        # channels_per_group = xh1[0].shape[2] // 3
        # s1, s2, s3 = torch.split(xh1[0], channels_per_group, 2)
        # sy1 = self.hconv1(s1.view(b,-1,h,w))
        # sy2 = self.hconv2(s2.view(b,-1,h,w))
        # sy3 = self.hconv3(s3.view(b,-1,h,w))

        # channels_per_group = xh1[0].shape[2] // 3
        # s1, s2, s3 = torch.split(xh1[0], channels_per_group, 2)
        #
        # b,c,n,h,w = xh1[0].shape
        # sy1 = self.hconv2(s1.view(b,-1,h,w))
        # sy2 = self.hconv2(s2.view(b,-1,h,w))
        # sy3 = self.hconv2(s3.view(b,-1,h,w))
        #
        #
        # yh1 = torch.stack(
        #     [sy1, sy2, sy3], dim=2
        # )
        # # #
        # yh1_ = [None]
        # yh1_[0] = yh1
        #
        # y = self.WTI((xl1, yh1_))
        #y = self.conv(y1)
        # out = F.interpolate(yl1,scale_factor=2)


        # yh1 = self.hconv1(torch.squeeze(torch.mean((xh[0]), dim=2), 2))
        # yh2 = self.hconv2(torch.squeeze(torch.mean((xh[1]), dim=2), 2))
        # yh3 = self.hconv3(torch.squeeze(torch.mean((xh[2]), dim=2), 2))
        # # yh4 = self.hconv4(torch.squeeze(torch.mean((xh[3]), dim=2), 2))
        # yh = [xh[0] * torch.unsqueeze(yh1, 2), xh[1] * torch.unsqueeze(yh2, 2), xh[2] * torch.unsqueeze(yh3, 2)]

        #

        #out = torch.concat((transy,convsy),1)
        out = self.fusion(transy,convsy)


        # avg_attn = torch.mean(out, dim=1, keepdim=True)
        # max_attn, _ = torch.max(out, dim=1, keepdim=True)
        # agg = torch.cat([avg_attn, max_attn], dim=1)
        # sig = self.conv_squeeze(agg).sigmoid()
        #
        # attn = transy * sig[:, 0, :, :].unsqueeze(1) + convsy * sig[:, 1, :, :].unsqueeze(1)
        # attn = self.conv(attn)
        # return x * attn

        #
        return out

# class CWT3(nn.Module):
#     def __init__(self, in_dim, out_dim, wave, level):
#         super(CWT3, self).__init__()
#
#         # pre = torch.randn(1,1,size,size)
#         self.WTF = DWTForward(J=level, wave=wave)
#         self.WTI = DWTInverse(mode='zero', wave=wave)
#         # xl,xh = self.WTF(pre)
#         self.lconv = DEPTHWISECONV(in_dim,out_dim,1)
#         # self.hconv1 = GConv2(int(in_dim/4),int(out_dim/4),5,nScale=5,padding=2)
#         # self.hconv2 = GConv2(int(in_dim/4),int(out_dim/4),7,nScale=7,padding=3)
#
#         # self.hconv3 = DEPTHWISECONV(in_dim, out_dim, kernel=9)
#         # self.hconv4 = DEPTHWISECONV(in_dim, out_dim, kernel=11)
#
#     def forward(self, x):
#         WTF = self.WTF
#         WTI = self.WTI
#
#         xl, xh = WTF(x)
#         yl = self.lconv(xl)
#         # yh1 = self.hconv1(torch.squeeze(torch.mean((xh[0]), dim=2), 2))
#         # yh2 = self.hconv2(torch.squeeze(torch.mean((xh[1]), dim=2), 2))
#         # # yh3 = self.hconv3(torch.squeeze(torch.mean((xh[2]), dim=2), 2))
#         # # yh4 = self.hconv4(torch.squeeze(torch.mean((xh[3]), dim=2), 2))
#         # yh = [xh[0] * torch.unsqueeze(yh1, 2), xh[1] * torch.unsqueeze(yh2, 2)]
#         wt_out = WTI((yl, xh))
#
#         return wt_out
# class CWT4(nn.Module):
#     def __init__(self, in_dim, out_dim, wave, level):
#         super(CWT4, self).__init__()
#
#         self.WTF = DWTForward(J=level, wave=wave)
#         self.WTI = DWTInverse(mode='zero', wave=wave)
#         # xl,xh = self.WTF(pre)
#         self.lconv = DEPTHWISECONV(in_dim,out_dim,1)
#         # self.hconv1 = GConv2(int(in_dim/4),int(out_dim/4),5,nScale=5,padding=2)
#
#
#         # self.hconv2 = DEPTHWISECONV(in_dim, out_dim, kernel=7)
#         # self.hconv3 = DEPTHWISECONV(in_dim, out_dim, kernel=9)
#         # self.hconv4 = DEPTHWISECONV(in_dim, out_dim, kernel=11)
#
#     def forward(self, x):
#         WTF = self.WTF
#         WTI = self.WTI
#
#         xl, xh = WTF(x)
#         yl = self.lconv(xl)
#         # yh1 = self.hconv1(torch.squeeze(torch.mean((xh[0]), dim=2), 2))
#         # # yh2 = self.hconv2(torch.squeeze(torch.mean((xh[1]), dim=2), 2))
#         # # yh3 = self.hconv3(torch.squeeze(torch.mean((xh[2]), dim=2), 2))
#         # # yh4 = self.hconv4(torch.squeeze(torch.mean((xh[3]), dim=2), 2))
#         # yh = [xh[0] * torch.unsqueeze(yh1, 2)]
#         wt_out = WTI((yl, xh))
#
#         return wt_out

# class ABCNet(nn.Module):
#     def __init__(self, in_ch=3, out_ch=1, dim=64, ori_h=256, deep_supervision=True, **kwargs):
#         super(ABCNet, self).__init__()
#         self.deep_supervision = deep_supervision
#         filters = [dim, dim * 2, dim * 4, dim * 8, dim * 16]
#         features = [ori_h // 2, ori_h // 4, ori_h // 8, ori_h // 16]
#         self.maxpools = nn.ModuleList([nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(4)])
#         self.Conv1 = conv_block(in_ch=in_ch, out_ch=filters[0])
#         # self.Conv1 = ConvTransformer(in_ch, filters[0], pow(ori_h, 2), ori_h)
#         self.Convtans2 = ConvTransformer(filters[0], filters[1], pow(features[0], 2), features[0])
#         self.Convtans3 = ConvTransformer(filters[1], filters[2], pow(features[1], 2), features[1])
#         self.Convtans4 = ConvTransformer(filters[2], filters[3], pow(features[2], 2), features[2])
#         self.Conv5 = dconv_block(in_ch=filters[3], out_ch=filters[4])
#
#         self.Up5 = up_conv(filters[4], filters[3])
#         self.Up_conv5 = dconv_block(filters[4], filters[3])
#
#         self.Up4 = up_conv(filters[3], filters[2])
#         self.Up_conv4 = conv_block(filters[3], filters[2])
#
#         self.Up3 = up_conv(filters[2], filters[1])
#         self.Up_conv3 = conv_block(filters[2], filters[1])
#
#         self.Up2 = up_conv(filters[1], filters[0])
#         self.Up_conv2 = conv_block(filters[1], filters[0])
#
#         self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
#
#         # --------------------------------------------------------------------------------------------------------------
#         self.conv5 = nn.Conv2d(filters[4], out_ch, kernel_size=3, stride=1, padding=1)
#         self.conv4 = nn.Conv2d(filters[3], out_ch, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(filters[2], out_ch, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(filters[1], out_ch, kernel_size=3, stride=1, padding=1)
#         self.conv1 = nn.Conv2d(filters[0], out_ch, kernel_size=3, stride=1, padding=1)
#         # --------------------------------------------------------------------------------------------------------------
#
#     def forward(self, x):  # x(1,3,256,256)
#         e1 = self.Conv1(x)  # e1(1,64,256,256)
#         e2 = self.maxpools[0](e1)  # e2(1,64,128,128)
#         e2 = self.Convtans2(e2)  # e2(1,128,128,128)
#
#         e3 = self.maxpools[1](e2)  # e3(1,128,64,64)
#         e3 = self.Convtans3(e3)  # e3(1,256,64,64)
#
#         e4 = self.maxpools[2](e3)  # e4(1,256,32,32)
#         e4 = self.Convtans4(e4)  # e4(1,512,32,32)
#
#         e5 = self.maxpools[3](e4)  # e5(1,512,16,16)
#         e5 = self.Conv5(e5)  # e5(1,1024,16,16)
#
#         d5 = self.Up5(e5)  # d5(1,512,32,32)
#         d5 = torch.cat((e4, d5), dim=1)  # d5(1,1024,32,32)
#         d5 = self.Up_conv5(d5)  # d5(1,512,32,32)
#
#         d4 = self.Up4(d5)  # d4(1,256,64,64)
#         d4 = torch.cat((e3, d4), dim=1)  # d4(1,512,64,64)
#         d4 = self.Up_conv4(d4)  # d4(1,256,64,64)
#
#         d3 = self.Up3(d4)  # d3(1,128,128,128)
#         d3 = torch.cat((e2, d3), dim=1)  # d3(1,256,128,128)
#         d3 = self.Up_conv3(d3)  # d3(1,128,128,128)
#
#         d2 = self.Up2(d3)  # d2(1,64,256,256)
#         d2 = torch.cat((e1, d2), dim=1)  # d2(1,128,256,256)
#         d2 = self.Up_conv2(d2)  # d2(1,64,256,256)
#
#         out = self.Conv(d2)  # out(1,1,256,256)
#
#         d_s1 = self.conv1(d2)  # d_s1(1,1,256,256)
#         d_s2 = self.conv2(d3)  # d_s2(1,1,128,128)
#         d_s2 = _upsample_like(d_s2, d_s1)  # d_s2(1,1,256,256)
#         d_s3 = self.conv3(d4)  # d_s3(1,1,64,64)
#         d_s3 = _upsample_like(d_s3, d_s1)  # d_s3(1,1,256,256)
#         d_s4 = self.conv4(d5)  # d_s4(1,1,32,32)
#         d_s4 = _upsample_like(d_s4, d_s1)  # d_s4(1,1,256,256)
#         d_s5 = self.conv5(e5)  # d_s5(1,1,16,16)
#         d_s5 = _upsample_like(d_s5, d_s1)  # d_s5(1,1,256,256)
#         if self.deep_supervision:
#             outs = [d_s1, d_s2, d_s3, d_s4, d_s5, out]  # d_s1(1,1,256,256) d_s2(1,1,256,256) d_s3(1,1,256,256) d_s4(1,1,256,256) d_s5(1,1,256,256)
#         else:
#             outs = out
#         # d1 = self.active(out)
#
#         return outs
#
#
# if __name__ == '__main__':
#     x = torch.randn(1, 64, 160, 160)
#     model = CWT1(64,64,160)
#     print(model(x))


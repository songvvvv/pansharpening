import torch.nn as nn
import torch.nn.functional as F

import math
import torch
from torch import nn
from einops.layers.torch import Rearrange
import models.cddfuse
from models.cddfuse import TransformerBlock

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        # B, C, H, W = x.shape
        # x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        # pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        # x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        # x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        x2 = torch.cat([x, pattn1], dim=1)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2
class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def get_weight(self):
        conv_weight = self.conv.weight  # 获取标准卷积层的权重
        conv_shape = conv_weight.shape  # 获取权重的形状
        # 将权重进行重排，将卷积核中的空间维度展平到通道维度
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        # 创建一个与标准权重相同形状的全零张量
        conv_weight_cd = torch.zeros(conv_shape[0], conv_shape[1], 3 * 3, device=conv_weight.device)  ####.cuda
        # conv_weight_cd = torch.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0).cuda   ####.cuda
        # 将重排后的权重复制到全零张量的对应位置
        conv_weight_cd[:, :, :] = conv_weight[:, :, :]
        # 在中心位置的权重值减去其他位置的权重值的和，实现条件卷积
        conv_weight_cd[:, :, 4] = conv_weight[:, :, 4] - conv_weight[:, :, :].sum(2)
        # 将处理后的权重恢复原来的形状
        conv_weight_cd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(
            conv_weight_cd)
        return conv_weight_cd, self.conv.bias



class Conv2d_ad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_ad, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_ad = conv_weight - self.theta * conv_weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        conv_weight_ad = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(
            conv_weight_ad)
        return conv_weight_ad, self.conv.bias


class Conv2d_rd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=2, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_rd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):

        if math.fabs(self.theta - 0.0) < 1e-8:
            out_normal = self.conv(x)
            return out_normal
        else:
            conv_weight = self.conv.weight
            conv_shape = conv_weight.shape
            if conv_weight.is_cuda:
                conv_weight_rd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 5 * 5).fill_(0)
            else:
                conv_weight_rd = torch.zeros(conv_shape[0], conv_shape[1], 5 * 5)
            conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
            conv_weight_rd[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = conv_weight[:, :, 1:]
            conv_weight_rd[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -conv_weight[:, :, 1:] * self.theta
            conv_weight_rd[:, :, 12] = conv_weight[:, :, 0] * (1 - self.theta)
            conv_weight_rd = conv_weight_rd.view(conv_shape[0], conv_shape[1], 5, 5)
            out_diff = nn.functional.conv2d(input=x, weight=conv_weight_rd, bias=self.conv.bias,
                                            stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups)

            return out_diff


class Conv2d_hd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_hd, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        # conv_weight_hd = torch.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0).cuda  ####.cuda
        conv_weight_hd = torch.zeros(conv_shape[0], conv_shape[1], 3 * 3, device=conv_weight.device)
        conv_weight_hd[:, :, [0, 3, 6]] = conv_weight[:, :, :]
        conv_weight_hd[:, :, [2, 5, 8]] = -conv_weight[:, :, :]
        conv_weight_hd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(
            conv_weight_hd)
        return conv_weight_hd, self.conv.bias


class Conv2d_vd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_vd, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_vd = torch.zeros(conv_shape[0], conv_shape[1], 3 * 3, device=conv_weight.device)
        # conv_weight_vd = torch.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0).cuda ####.cuda
        conv_weight_vd[:, :, [0, 1, 2]] = conv_weight[:, :, :]
        conv_weight_vd[:, :, [6, 7, 8]] = -conv_weight[:, :, :]
        conv_weight_vd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(
            conv_weight_vd)
        return conv_weight_vd, self.conv.bias


class DEConv(nn.Module):
    def __init__(self, dim):
        super(DEConv, self).__init__()
        self.conv1_1 = Conv2d_cd(dim, dim, 3, bias=True)
        self.conv1_2 = Conv2d_hd(dim, dim, 3, bias=True)
        self.conv1_3 = Conv2d_vd(dim, dim, 3, bias=True)
        self.conv1_4 = Conv2d_ad(dim, dim, 3, bias=True)
        self.conv1_5 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)

    def forward(self, x):
        w1, b1 = self.conv1_1.get_weight()
        w2, b2 = self.conv1_2.get_weight()
        w3, b3 = self.conv1_3.get_weight()
        w4, b4 = self.conv1_4.get_weight()
        w5, b5 = self.conv1_5.weight, self.conv1_5.bias

        w = w1 + w2 + w3 + w4 + w5
        b = b1 + b2 + b3 + b4 + b5
        res = nn.functional.conv2d(input=x, weight=w, bias=b, stride=1, padding=1, groups=1)

        return res

# class DEABlockTrain(nn.Module):
#     def __init__(self, conv, dim, kernel_size, reduction=8):
#         super(DEABlockTrain, self).__init__()
#         #self.conv1 = DEConv(dim)
#         self.conv1 = conv(dim, dim, kernel_size, bias=True)
#         self.act1 = nn.ReLU(inplace=True)
#         self.conv2 = conv(dim, dim, kernel_size, bias=True)
#         self.sa = SpatialAttention()
#         self.ca = ChannelAttention(dim, reduction)
#         self.pa = PixelAttention(dim)
#
#     def forward(self, x):
#         res = self.conv1(x)
#         res = self.act1(res)
#         res = res + x
#         res = self.conv2(res)
#         cattn = self.ca(res)
#         sattn = self.sa(res)
#         pattn1 = sattn + cattn
#         pattn2 = self.pa(res, pattn1)
#         res = res * pattn2
#         res = res + x
#         return res
class DEBlockTrain(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(DEBlockTrain, self).__init__()
        #self.conv1 = DEConv(dim)
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.act1(res)
        res = res + x
        res = self.conv2(res)
        res = res + x
        return res
class DEBlockTrain_PAN(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(DEBlockTrain_PAN, self).__init__()
        #self.conv1 = DEConv(dim)
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.conv1_1 = Conv2d_cd(dim, dim, 3, bias=True)
        self.conv1_2 = Conv2d_hd(dim, dim, 3, bias=True)
        self.conv1_3 = Conv2d_vd(dim, dim, 3, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)

    def forward(self, x):
        w1, b1 = self.conv1_1.get_weight()
        w2, b2 = self.conv1_2.get_weight()
        w3, b3 = self.conv1_3.get_weight()

        res = self.conv1(x)
        res_1 = nn.functional.conv2d(input=x, weight=w1, bias=b1, stride=1, padding=1)
        res_2 = nn.functional.conv2d(input=x, weight=w2, bias=b2, stride=1, padding=1)
        res_3 = nn.functional.conv2d(input=x, weight=w3, bias=b3, stride=1, padding=1)
        res = res + res_1+ res_2+ res_3
        res = self.act1(res)
        res = res + x
        res = self.conv2(res)
        res = res + x
        return res


class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, m, p, z):
        #initial = x + y
        cattn = self.ca(z)
        sattn = self.sa(z)
        pattn1 = sattn + cattn
        pattn2_m = self.sigmoid(self.pa(m, pattn1))
        pattn2_p = self.sigmoid(self.pa(p, pattn1))
        result = z + pattn2_m * m + pattn2_p * p
        result = self.conv(result)
        return result

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class DEANet(nn.Module):
    def __init__(self, base_dim=32):
        super(DEANet, self).__init__()
        # down-sample
        self.down1_m = nn.Sequential(nn.Conv2d(4, base_dim, kernel_size=1, stride = 1, padding=0))
        self.down2_m = nn.Sequential(nn.Conv2d(base_dim, base_dim, kernel_size=3, stride=2, padding=1),  #尺寸减半
                                   nn.ReLU(True))
        self.down3_m = nn.Sequential(nn.Conv2d(base_dim, base_dim, kernel_size=3, stride=2, padding=1), #尺寸减半
                                   nn.ReLU(True))
        self.down1_p = nn.Sequential(nn.Conv2d(1, base_dim, kernel_size=1, stride=1, padding=0))
        self.down2_p = nn.Sequential(nn.Conv2d(base_dim, base_dim , kernel_size=3, stride=2, padding=1),  # 尺寸减半
                                    nn.ReLU(True))
        self.down3_p = nn.Sequential(nn.Conv2d(base_dim , base_dim , kernel_size=3, stride=2, padding=1),  # 尺寸减半
                                    nn.ReLU(True))
        self.down1_pm = nn.Sequential(nn.Conv2d(5, base_dim, kernel_size=1, stride=1, padding=0))
        self.down2_pm = nn.Sequential(nn.Conv2d(base_dim, base_dim , kernel_size=3, stride=2, padding=1),  # 尺寸减半
                                     nn.ReLU(True))
        self.down3_pm = nn.Sequential(nn.Conv2d(base_dim, base_dim , kernel_size=3, stride=2, padding=1),  # 尺寸减半
                                     nn.ReLU(True))
        self.down1_pm_t = nn.Sequential(nn.Conv2d(5, base_dim, kernel_size=1, stride=1, padding=0))
        self.down2_pm_t = nn.Sequential(nn.Conv2d(base_dim, base_dim, kernel_size=3, stride=2, padding=1),  # 尺寸减半
                                      nn.ReLU(True))
        self.down3_pm_t = nn.Sequential(nn.Conv2d(base_dim, base_dim, kernel_size=3, stride=2, padding=1),  # 尺寸减半
                                      nn.ReLU(True))
        # level1
        self.down_level1_block1_m = DEBlockTrain(default_conv, base_dim, 3)
        self.down_level1_block2_m = DEBlockTrain(default_conv, base_dim, 3)


        self.down_level1_block1_p = DEBlockTrain(default_conv, base_dim, 3)
        self.down_level1_block2_p = DEBlockTrain(default_conv, base_dim, 3)


        self.down_level1_block1_pm = DEBlockTrain(default_conv, base_dim, 3)
        self.down_level1_block2_pm = DEBlockTrain(default_conv, base_dim, 3)


        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=base_dim, num_heads=8, ffn_expansion_factor=2,
                               bias=False, LayerNorm_type='WithBias') for i in range(4)])
        self.up_level1_block1 = DEBlockTrain(default_conv, base_dim, 3)
        self.up_level1_block2 = DEBlockTrain(default_conv, base_dim, 3)

        # level2
        self.fe_level_2_m = nn.Conv2d(in_channels=base_dim + 4, out_channels=base_dim, kernel_size=1, stride=1, padding=0)
        self.down_level2_block1_m = DEBlockTrain(default_conv, base_dim, 3)
        self.down_level2_block2_m = DEBlockTrain(default_conv, base_dim, 3)

        self.fe_level_2_p = nn.Conv2d(in_channels=base_dim + 1, out_channels=base_dim, kernel_size=1, stride=1,padding=0)
        self.down_level2_block1_p = DEBlockTrain(default_conv, base_dim, 3)
        self.down_level2_block2_p = DEBlockTrain(default_conv, base_dim, 3)


        self.fe_level_2_pm = nn.Conv2d(in_channels=base_dim, out_channels=base_dim , kernel_size=1, stride=1,padding=0)
        self.down_level2_block1_pm = DEBlockTrain(default_conv, base_dim, 3)
        self.down_level2_block2_pm = DEBlockTrain(default_conv, base_dim, 3)



        self.encoder_level2 = nn.Sequential(
            *[TransformerBlock(dim=base_dim , num_heads=8, ffn_expansion_factor=2,
                               bias=False, LayerNorm_type='WithBias') for i in range(4)])
        self.up_level2_block1 = DEBlockTrain(default_conv, base_dim, 3)
        self.up_level2_block2 = DEBlockTrain(default_conv, base_dim, 3)


        # level3
        self.fe_level_3_m = nn.Conv2d(in_channels=base_dim + 4, out_channels=base_dim, kernel_size=1, stride=1, padding=0)
        self.down_level3_block1_m = DEBlockTrain(default_conv, base_dim , 3)
        self.down_level3_block2_m = DEBlockTrain(default_conv, base_dim, 3)


        self.fe_level_3_p = nn.Conv2d(in_channels=base_dim + 1, out_channels=base_dim, kernel_size=1, stride=1,padding=0)
        self.down_level3_block1_p = DEBlockTrain(default_conv, base_dim, 3)
        self.down_level3_block2_p = DEBlockTrain(default_conv, base_dim, 3)


        self.fe_level_3_pm = nn.Conv2d(in_channels=base_dim , out_channels=base_dim , kernel_size=1, stride=1,
                                      padding=0)
        self.down_level3_block1_pm = DEBlockTrain(default_conv, base_dim, 3)
        self.down_level3_block2_pm = DEBlockTrain(default_conv, base_dim, 3)



        self.encoder_level3 = nn.Sequential(
            *[TransformerBlock(dim=base_dim, num_heads=8, ffn_expansion_factor=2,
                               bias=False, LayerNorm_type='WithBias') for i in range(4)])
        self.level3_block1 = DEBlockTrain(default_conv, base_dim, 3)
        self.level3_block2 = DEBlockTrain(default_conv, base_dim, 3)




        # up-sample
        self.up1 = nn.Sequential(nn.ConvTranspose2d(base_dim, base_dim, kernel_size=3, stride=2, padding=1, output_padding=1),  # 尺寸2倍
                                 nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(base_dim, base_dim, kernel_size=3, stride=2, padding=1, output_padding=1),  # 尺寸2倍
                                 nn.ReLU(True))
        self.up3 = nn.Sequential(nn.Conv2d(base_dim, 4, kernel_size=1, stride=1, padding=0))
        # feature fusion
        self.mix3 = CGAFusion(base_dim)
        self.mix2 = CGAFusion(base_dim)
        self.mix1 = CGAFusion(base_dim)
    def forward(self, m, p):
        m1 = F.interpolate(m, scale_factor=0.5, mode="bicubic")
        m2 = F.interpolate(m1, scale_factor=0.5, mode="bicubic")
        p1 = F.interpolate(p, scale_factor=0.5, mode="bicubic")
        p2 = F.interpolate(p1, scale_factor=0.5, mode="bicubic")
        pm = torch.cat((p, m), dim=1)

        m_down1 = self.down1_m(m)
        m_down1 = self.down_level1_block1_m(m_down1)
        m_down1 = self.down_level1_block2_m(m_down1)


        p_down1 = self.down1_p(p)
        p_down1 = self.down_level1_block1_p(p_down1)
        p_down1 = self.down_level1_block2_p(p_down1)


        pm_down1 = self.down1_pm(pm)
        pm_down1_init = self.down_level1_block1_pm(pm_down1)
        pm_down1_init = self.down_level1_block2_pm(pm_down1_init)


        pm_down1_t = self.down1_pm_t(pm)
        pm_trans1 = self.encoder_level1(pm_down1_t)
        pm_down1_add = pm_trans1 + pm_down1_init

        m_down2 = self.down2_m(m_down1)
        m_down2_ = torch.cat((m_down2, m1), dim=1)
        m_down2_init = self.fe_level_2_m(m_down2_)
        m_down2_init = self.down_level2_block1_m(m_down2_init)
        m_down2_init = self.down_level2_block2_m(m_down2_init)


        p_down2 = self.down2_p(p_down1)
        p_down2_ = torch.cat((p_down2, p1), dim=1)
        p_down2_init = self.fe_level_2_p(p_down2_)
        p_down2_init = self.down_level2_block1_p(p_down2_init)
        p_down2_init = self.down_level2_block2_p(p_down2_init)

        pm_down2 = self.down2_pm(pm_down1_add)
        pm_down2_init = self.fe_level_2_pm(pm_down2)
        pm_down2_init = self.down_level2_block1_pm(pm_down2_init)
        pm_down2_init = self.down_level2_block2_pm(pm_down2_init)


        pm_trans2 = self.down2_pm_t(pm_down1_add)
        pm_trans2 = self.encoder_level2(pm_trans2)
        pm_down2_add = pm_trans2 + pm_down2_init


        m_down3 = self.down3_m(m_down2_init)
        m_down3_ = torch.cat((m_down3, m2), dim=1)
        m_down3_init = self.fe_level_3_m(m_down3_)
        m_down3_init = self.down_level3_block1_m(m_down3_init)
        m_down3_init = self.down_level3_block2_m(m_down3_init)

        p_down3 = self.down3_p(p_down2_init)
        p_down3_ = torch.cat((p_down3, p2), dim=1)
        p_down3_init = self.fe_level_3_p(p_down3_)
        p_down3_init = self.down_level3_block1_p(p_down3_init)
        p_down3_init = self.down_level3_block2_p(p_down3_init)

        pm_down3 = self.down3_pm(pm_down2_add)
        pm_down3_init = self.fe_level_3_pm(pm_down3)
        pm_down3_init = self.down_level3_block1_pm(pm_down3_init)
        pm_down3_init = self.down_level3_block2_pm(pm_down3_init)


        pm_trans3 = self.down3_pm_t(pm_down2_add)
        pm_trans3 = self.encoder_level3(pm_trans3)
        pm_down3_add = pm_trans3 + pm_down3_init


        x_level3_mix = self.mix3(m_down3_init, p_down3_init, pm_down3_add)
        x1 = self.level3_block1(x_level3_mix)
        x2 = self.level3_block2(x1)



        x_up1 = self.up1(x2)
        x_level2_mix = self.mix2(m_down2_init, p_down2_init, x_up1)
        x_up1 = self.up_level2_block1(x_level2_mix)
        x_up1 = self.up_level2_block2(x_up1)




        x_up2 = self.up2(x_up1)
        x_level1_mix = self.mix1(m_down1, p_down1, x_up2)
        x_up2 = self.up_level1_block1(x_level1_mix)
        x_up2 = self.up_level1_block2(x_up2)


        out = self.up3(x_up2)

        return out
from thop import profile
import time
if __name__ == '__main__':
    xms = torch.randn(1, 4, 256, 256)
    pan = torch.randn(1, 1, 256, 256)
    Model = DEANet()
    start = time.perf_counter()
    #for i in range(100):
    Y = Model(xms, pan)
    end = time.perf_counter()
    #time = (end - start)/100
    print(f"Elapsed time: {end - start:.4f} seconds")
    flops, params = profile(Model, inputs=(xms, pan), verbose=0)
    print('FLOPs=' + str(flops / 1e9) + 'G')
    print('params=' + str(params / 1e3) + 'K')
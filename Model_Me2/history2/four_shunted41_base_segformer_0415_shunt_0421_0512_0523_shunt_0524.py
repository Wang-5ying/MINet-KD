import importlib
from math import sqrt
import time
import numpy as np
import torch
from mmseg.models import build_segmentor
from torch import nn
import torch.nn.functional as F
# from MLNet.ResT.models.rest_v2 import ResTV2

# from codes.bayibest82segformerbest.External_Attention.model.attention.ExternalAttention import ExternalAttention
# from bayibestsegformer.baseapi.newapii711715 import BasicConv2d, node, CA, GM2  # atttention
# from MLNet.mobilevit import mobilevit_s
from mmcv import Config
from torchvision.models import vgg16, vgg16_bn
from collections import OrderedDict
from plug_and_play_modules.DO_Conv.do_conv_pytorch_1_10 import DOConv2d
from timm.models.layers import DropPath, trunc_normal_
from mmseg.models.backbones.mix_transformer import mit_b5
from timm.models import create_model
from codes.GCoNet_plus_For_Four_Model.Model_Me.decoders.MLPDecoder import DecoderHead
#############630 qudiaol dongtaijuanji   vt5000 0.026 150epoch
from codes.GCoNet_plus_For_Four_Model.Model_Me.module import DD, SpatialAttention, CM, BasicConv2d, SELayer, ASPP, MixerBlock, SpatialAttention, CA, MLP
# from backbone.Shunted_Transformer.SSA import shunted_b
# from mmseg.models.backbones.mix_transformer import mit_b4
from backbone.Shunted_Transformer.SSA import shunted_b
from 文献代码.FcaNet.model.layer import MultiSpectralAttentionLayer
from codes.GCoNet.models.GCoNet import AllAttLayer
def shuffle(x):
    B, C, H, W = x.shape
    idx = torch.randperm(C)
    x = x[idx, :, :, :].contiguous
    return x
class MAttention(nn.Module):
    def __init__(self, dim, reduction=8, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv_r = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_f = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, rgb):
        B, C, H, W = rgb.shape
        rgb = rgb.reshape(B, H * W, C)
        B, N, C = rgb.shape

        qkv_r = self.qkv_r(rgb).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qr, kr, vr = qkv_r[0], qkv_r[1], qkv_r[2]  # make torchscript happy (cannot use tensor as tuple)
        qkv_g = self.qkv_f(rgb).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qg, kg, vg = qkv_g[0], qkv_g[1], qkv_g[2]  # make torchscript happy (cannot use tensor as tuple)
        attn_r = (qr @ kg.transpose(-2, -1)) * self.scale
        # attn_r = (qf @ kr) * self.scale
        attn_r = attn_r.softmax(dim=-1)
        attn_r = self.attn_drop(attn_r)
        rgb_a = (attn_r @ vg).transpose(1, 2).reshape(B, N, C)
        rgb_a = self.proj(rgb_a)
        rgb_a = self.proj_drop(rgb_a)

        B, N, C = rgb_a.shape
        rgb_a = rgb_a.reshape(B, C, int(sqrt(N)), int(sqrt(N)))
        # print(rgb_a.size())
        return rgb_a
class CAAM_WBY(nn.Module):
    """
    Class Activation Attention Module
    """

    def __init__(self, feat_in, num_classes, bin_size, norm_layer):
        super(CAAM_WBY, self).__init__()
        feat_inner = feat_in // 2
        self.norm_layer = norm_layer
        self.bin_size = bin_size
        # self.dropout = nn.Dropout2d(0.1)
        # 1
        self.conv_cam = nn.Conv2d(feat_in, num_classes, kernel_size=1)
        self.pool_cam = nn.AdaptiveAvgPool2d(bin_size)
        self.sigmoid = nn.Sigmoid()
        # 2
        self.conv_cam_y = nn.Conv2d(feat_in, num_classes, kernel_size=1)
        self.pool_cam_y = nn.AdaptiveAvgPool2d(bin_size)

        bin_num = bin_size[0] * bin_size[1]
        # 1

        self.fuse = nn.Conv2d(bin_num, 1, kernel_size=1)

        # 2

        self.fuse_y = nn.Conv2d(bin_num, 1, kernel_size=1)

        # 1
        self.conv_out = nn.Sequential(
            nn.Conv2d(feat_in, feat_in, kernel_size=1, bias=False),
            norm_layer(feat_in),
            nn.ReLU(inplace=True)
        )
        self.scale = feat_inner ** -0.5
        self.relu = nn.ReLU(inplace=True)
        self.msa = MAttention(feat_in)
        # 2
        self.conv_out_y = nn.Sequential(
            nn.Conv2d(feat_in, feat_in, kernel_size=1, bias=False),
            norm_layer(feat_in),
            nn.ReLU(inplace=True)
        )
        self.relu_y = nn.ReLU(inplace=True)
        self.msa_y = MAttention(feat_in)
        # print("feat_in", feat_in)

    def forward(self, x, y):
        # print("x", x.size())
        ### 1
        residule = x
        cam1 = self.conv_cam(x)  # [B, K, H, W]
        cam = cam1
        cls_score = self.sigmoid(self.pool_cam(cam))  # [B, K, bin_num_h, bin_num_w]
        ms = self.msa(residule)
        out = self.conv_out(ms)  # 1017 去掉残差  + residule   🐶

        ### 2
        residule_y = y
        cam1_y = self.conv_cam_y(y)  # [B, K, H, W]
        cam_y = cam1_y
        cls_score_y = self.sigmoid(self.pool_cam_y(cam_y))  # [B, K, bin_num_h, bin_num_w]

        ms_y = self.msa_y(residule_y)
        out_y = self.conv_out_y(ms_y)  # 1017 去掉残差  + residule   🐶
        out = out + out_y
        cls_score = cls_score + cls_score_y
        return out, cls_score

class ResBlk(nn.Module):
    def __init__(self, channel_in=64, channel_out=64):
        super(ResBlk, self).__init__()
        self.conv_in = nn.Conv2d(channel_in, 64, 3, 1, 1)
        self.relu_in = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv2d(64, channel_out, 3, 1, 1)
        self.bn_in = nn.BatchNorm2d(64)
        self.bn_out = nn.BatchNorm2d(channel_out)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu_in(x)
        x = self.conv_out(x)
        x = self.bn_out(x)
        return x
class com(nn.Module):
    def __init__(self, inchannel1, inchannel2, size, size2, padding, dialation):
        super(com, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Linear(inchannel1, inchannel2)
        self.resn = ResBlk(inchannel2, inchannel2)
        self.zj = DD(inchannel1, inchannel2, size, size2, padding, dialation)
    def forward(self, high, now):
        weight = self.avg(high).flatten(1)
        weight = self.mlp(weight).unsqueeze(-1).unsqueeze(-1)
        res = weight * self.zj(high, now)
        res = self.resn(res)
        return res
class Me(nn.Module):
    def load_pre(self, pre_model1):
        new_state_dict3 = OrderedDict()
        state_dict = torch.load(pre_model1)
        for k, v in state_dict.items():
            name = k
            new_state_dict3[name] = v
        self.resnet.load_state_dict(new_state_dict3, strict=False)
        self.resnet2.load_state_dict(new_state_dict3, strict=False)
        self.resnet3.load_state_dict(new_state_dict3, strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model1}")
        print(f"Depth SwinTransformer loading pre_model ${pre_model1}")

    def __init__(self, mode='small'):
        super(Me, self).__init__()
        # self.resnet = shunted_b()
        # self.resnet2 = shunted_b()
        # self.resnet3 = shunted_b()
        self.resnet = shunted_b()
        self.resnet2 = shunted_b()
        self.resnet3 = shunted_b()
        self.config = Config()
        # self.ini = BasicConv2d(3, 256, 1)

        self.zc = BasicConv2d(12, 3, 1)

        self.cca1 = CA(64)
        self.cca2 = CA(64)
        self.csa1 = SpatialAttention(3)
        self.csa2 = SpatialAttention(3)
        self.cca3 = CA(256)
        self.cca4 = CA(256)
        self.csa3 = SpatialAttention(3)
        self.csa4 = SpatialAttention(3)

        self.sa1 = MultiSpectralAttentionLayer(64, 64, 64)
        self.sa2 = MultiSpectralAttentionLayer(128, 32, 32)
        self.sa3 = MultiSpectralAttentionLayer(256, 16, 16)
        self.sa4 = MultiSpectralAttentionLayer(512, 8, 8)

        self.re1 = nn.Conv2d(64 * 2, 64, 1)
        self.re2 = nn.Conv2d(128 * 2, 128, 1)
        self.re3 = nn.Conv2d(256 * 2, 256, 1)
        self.re4 = nn.Conv2d(512 * 2, 512, 1)

        self.sa1_2 = MultiSpectralAttentionLayer(64, 64, 64)
        self.sa2_2 = MultiSpectralAttentionLayer(128, 32, 32)
        self.sa3_2 = MultiSpectralAttentionLayer(256, 16, 16)
        self.sa4_2 = MultiSpectralAttentionLayer(512, 8, 8)

        self.doc = nn.Conv2d(512, 512, 1)
        self.prs0 = nn.Conv2d(512, 256, 1)
        self.prs1 = nn.Conv2d(256, 128, 1)
        self.prs2 = nn.Conv2d(128, 64, 1)

        self.DD1 = DD(512, 256, [4, 2], [8, 8], [3, 2], [2, 4])
        self.DD2 = DD(512, 128, [8, 4], [16, 16], [7, 3], [2, 2])
        self.DD3 = DD(512, 64, [16, 8], [32, 32], [15, 7], [2, 2])

        self.res0 = BasicConv2d(256, 1, 1)
        self.res1 = BasicConv2d(128, 1, 1)
        self.res2 = BasicConv2d(64, 1, 1)
        self.res3 = BasicConv2d(64, 1, 1)
        self.end = BasicConv2d(64, 1, 3, 1, 1)

        self.supself1 = BasicConv2d(64, 1, 1)

        # self.dct1 = MultiSpectralAttentionLayer(64, )
        self.aspp1 = ASPP(256, [7, 5, 3, 1], 256)
        self.aspp2 = ASPP(128, [7, 5, 3, 1], 128)
        self.aspp3 = ASPP(64, [7, 5, 3, 1], 64)

        self.mlp1 = MixerBlock(256, 128, 256) # 16 * 16
        self.mlp2 = MixerBlock(128, 64, 1024)
        self.mlp3 = MixerBlock(64, 32, 4096)
        self.mlp1_1 = MixerBlock(256, 128, 256)  # 16 * 16
        self.mlp2_1 = MixerBlock(128, 64, 1024)
        self.mlp3_1 = MixerBlock(64, 32, 4096)
        self.mlp_decode1 = DecoderHead([64, 128, 256, 512])

        self.coa1 = AllAttLayer(512)

        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.gap2 = nn.AdaptiveAvgPool2d(1)
        self.soft1 = nn.Softmax(dim=1)
        self.soft2 = nn.Softmax(dim=1)
        self.gap3 = nn.AdaptiveAvgPool2d(1)
        self.gap4 = nn.AdaptiveAvgPool2d(1)
        self.soft3 = nn.Softmax(dim=1)
        self.soft4 = nn.Softmax(dim=1)

        self.act = CAAM_WBY(512, 1, [32, 32], norm_layer=nn.BatchNorm2d)
    def forward(self, x, y):
        r = []
        d = []
        w = []
        B = x.shape[0]
        z = self.zc(torch.cat((x, y, x - y, x + y), dim=1))
        # ----------colomn 1--------------
        # r_stage1
        patch_embed = getattr(self.resnet, f"patch_embed{1}")
        block = getattr(self.resnet, f"block{1}")
        norm = getattr(self.resnet, f"norm{1}")
        x, H, W = patch_embed(x)
        for blk in block:
            x = blk(x, H, W)
        x = norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 64 * 64 *64
        r.append(x)

        # d_stage1
        patch_embed = getattr(self.resnet3, f"patch_embed{1}")
        block = getattr(self.resnet3, f"block{1}")
        norm = getattr(self.resnet3, f"norm{1}")
        y, H, W = patch_embed(y)
        for blk in block:
            y = blk(y, H, W)
        y = norm(y)
        y = y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 64 * 64 *64
        d.append(y)

        # z_stage1
        patch_embed = getattr(self.resnet2, f"patch_embed{1}")
        block = getattr(self.resnet2, f"block{1}")
        norm = getattr(self.resnet2, f"norm{1}")
        z, H, W = patch_embed(z)
        for blk in block:
            z = blk(z, H, W)
        z = norm(z)
        z = z.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 64 * 64 *64
        w.append(z)

        # ------- colomn2 ---------
        # z_stage2
        x = self.csa1(self.cca1(x + z)) + x
        y = self.csa2(self.cca2(y + z)) + y
        z = x + y + z
        z = torch.cat((x, y, z), dim=0)
        b, c, h, w1 = z.size()
        # z = self.zen1(z)
        patch_embed = getattr(self.resnet2, f"patch_embed{2}")
        block = getattr(self.resnet2, f"block{2}")
        norm = getattr(self.resnet2, f"norm{2}")
        z, h, w1 = patch_embed(z)
        for blk in block:
            z = blk(z, h, w1)
        z = norm(z)
        z = z.reshape(b, h, w1, -1).permute(0, 3, 1, 2).contiguous()  # 128 * 32 * 32
        b, c, h, w1 = z.size()
        x_he = z[0:int(b / 3), :, :, :]
        y_he = z[int(b / 3):2 * int(b / 3), :, :]
        z = z[2 * int(b / 3):b, :, :, :]

        # r_stage2
        patch_embed = getattr(self.resnet, f"patch_embed{2}")
        block = getattr(self.resnet, f"block{2}")
        norm = getattr(self.resnet, f"norm{2}")
        x, H, W = patch_embed(x)
        for blk in block:
            x = blk(x, H, W)
        x = norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        r.append(x)

        # d_stage2
        patch_embed = getattr(self.resnet3, f"patch_embed{2}")
        block = getattr(self.resnet3, f"block{2}")
        norm = getattr(self.resnet3, f"norm{2}")
        y, H, W = patch_embed(y)
        for blk in block:
            y = blk(y, H, W)
        y = norm(y)
        y = y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        d.append(y)

        # ------ colomn3------
        # x = x_he * x + x
        # y = y_he * y + x
        x = self.soft1(self.gap1(x_he)) * x + x
        y = self.soft2(self.gap2(y_he)) * y + y
        # r_stage3
        B, C, H, W = x.size()
        patch_embed = getattr(self.resnet, f"patch_embed{3}")
        block = getattr(self.resnet, f"block{3}")
        norm = getattr(self.resnet, f"norm{3}")
        x, H, W = patch_embed(x)
        for blk in block:
            x = blk(x, H, W)
        x = norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        r.append(x)

        # d_stage3
        patch_embed = getattr(self.resnet3, f"patch_embed{3}")
        block = getattr(self.resnet3, f"block{3}")
        norm = getattr(self.resnet3, f"norm{3}")
        y, H, W = patch_embed(y)
        for blk in block:
            y = blk(y, H, W)
        y = norm(y)
        y = y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        d.append(y)

        # z_stage3
        patch_embed = getattr(self.resnet2, f"patch_embed{3}")
        block = getattr(self.resnet2, f"block{3}")
        norm = getattr(self.resnet2, f"norm{3}")
        z, H, W = patch_embed(z)
        for blk in block:
            z = blk(z, H, W)
        z = norm(z)
        z = z.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        w.append(z)

        # ------- colomn4 ---------
        # z_stage4
        x = self.csa3(self.cca3(x + z)) + x
        y = self.csa4(self.cca4(y + z)) + y
        z = x + y + z
        z = torch.cat((x, y, z), dim=0)
        b, c, h, w1 = z.size()
        # z = self.zen1(z)
        patch_embed = getattr(self.resnet2, f"patch_embed{4}")
        block = getattr(self.resnet2, f"block{4}")
        norm = getattr(self.resnet2, f"norm{4}")
        z, h, w1 = patch_embed(z)
        for blk in block:
            z = blk(z, h, w1)
        z = norm(z)
        z = z.reshape(b, h, w1, -1).permute(0, 3, 1, 2).contiguous()  # 128 * 32 * 32
        b, c, h, w1 = z.size()
        x_he = z[0:int(b / 3), :, :, :]
        y_he = z[int(b / 3):2 * int(b / 3), :, :]
        z = z[2 * int(b / 3):b, :, :, :]

        # r_stage4
        patch_embed = getattr(self.resnet, f"patch_embed{4}")
        block = getattr(self.resnet, f"block{4}")
        norm = getattr(self.resnet, f"norm{4}")
        x, H, W = patch_embed(x)
        for blk in block:
            x = blk(x, H, W)
        x = norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        r.append(x)

        # d_stage4
        patch_embed = getattr(self.resnet3, f"patch_embed{4}")
        block = getattr(self.resnet3, f"block{4}")
        norm = getattr(self.resnet3, f"norm{4}")
        y, H, W = patch_embed(y)
        for blk in block:
            y = blk(y, H, W)
        y = norm(y)
        y = y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        d.append(y)

        # x = x * x_he + x
        # y = y * y_he + x
        x = self.soft3(self.gap3(x_he)) * x + x
        y = self.soft4(self.gap4(y_he)) * y + y
        r.append(x)
        d.append(y)



        rd = []

        re = self.re1(torch.cat((self.sa1(r[0]), self.sa1_2(d[0])), dim=1))
        rd.append(re)
        re = self.re2(torch.cat((self.sa2(r[1]), self.sa2_2(d[1])), dim=1))
        rd.append(re)
        re = self.re3(torch.cat((self.sa3(r[2]), self.sa3_2(d[2])), dim=1))
        rd.append(re)
        re = self.re4(torch.cat((self.sa4(r[3]), self.sa4_2(d[3])), dim=1))
        rd.append(re)
        # for i in rd:
        #     print("rd", i.size())
        rd[-1] = self.coa1(rd[-1])
        decode = []

        d0 = self.doc(r[-1] + d[-1])
        decode.append(d0)

        d0 = self.prs0(F.interpolate(d0, scale_factor=2))
        x = self.DD1(rd[-1], rd[-2])
        y = self.aspp1(d0)
        d1 = self.mlp1(x * y + x + y)
        d1 = self.mlp1_1(d1)
        decode.append(d1)

        d1 = self.prs1(F.interpolate(d1, scale_factor=2))
        x = self.DD2(rd[-1], rd[-3])
        y = self.aspp2(d1)
        d2 = self.mlp2(x * y + x + y)
        d2 = self.mlp2_1(d2)
        decode.append(d2)

        d2 = self.prs2(F.interpolate(d2, scale_factor=2))
        x = self.DD3(rd[-1],rd[-4])
        y = self.aspp3(d2)
        d3 = self.mlp3(x * y + x + y)
        d3 = self.mlp3_1(d3)
        decode.append(d3)

        decoder = [d0, d0, d0, d0]
        decoder[0] = decode[-1]
        decoder[1] = decode[-2]
        decoder[2] = decode[-3]
        decoder[3] = decode[-4]

        mlp_decode = F.interpolate(self.mlp_decode1(decoder), size=256)
        # print(d3.size(),"d3")
        res0 = F.interpolate(self.res0(d0), size=256)
        res1 = F.interpolate(self.res1(d1), size=256)
        res2 = F.interpolate(self.res2(d2), size=256)
        res3 = F.interpolate(self.res3(d3), size=256)
        res = F.interpolate(self.end(d3), size=256)
        returnvalues = []
        returnvalues.append(res0)
        returnvalues.append(res1)
        returnvalues.append(res2)
        returnvalues.append(res3)
        returnvalues.append(res)
        returnvalues.append(mlp_decode)

        # r
        returnr = []
        B, C, H, W = r[-2].size()
        rhigh = r[-2].view(B, H * W, C)
        # print(rhigh.size())
        rhigh = torch.matmul(rhigh, rhigh.permute(0, 2, 1)).view(B, 1, H * W, H * W)
        returnr.append(rhigh)

        rlow = self.supself1(r[0])
        returnr.append(rlow)

        # d
        returnd = []
        B, C, H, W = d[-2].size()
        dhigh = d[-2].view(B, H * W, C)
        dhigh = torch.matmul(dhigh, dhigh.permute(0, 2, 1)).view(B, 1, H * W, H * W)
        returnd.append(dhigh)

        dlow = self.supself1(d[0])
        returnd.append(dlow)
        # for i in returnd:
        #     print("d", i.size())
        # for i in returnr:
        #     print("r", i.size())
        act = self.act(r[-1], d[-1])
        if self.training:
            return returnvalues, returnr, returnd, act
        else:
            return returnvalues


if __name__ == '__main__':
    a = torch.randn(2, 3, 256, 256).cuda()
    b = torch.randn(2, 3, 256, 256).cuda()
    model = Me()
    model.cuda()
    # model.load_pre("/media/wby/shuju/ckpt_B.pth")
    out = model(a, b)
    # end = time.time()

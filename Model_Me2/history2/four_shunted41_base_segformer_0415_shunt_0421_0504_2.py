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

#############630 qudiaol dongtaijuanji   vt5000 0.026 150epoch
from codes.GCoNet_plus_For_Four_Model.Model_Me.module import DD, SpatialAttention, CM, BasicConv2d, SELayer, ASPP
# from backbone.Shunted_Transformer.SSA import shunted_b
# from mmseg.models.backbones.mix_transformer import mit_b5
from backbone.Shunted_Transformer.SSA import shunted_b
from 文献代码.FcaNet.model.layer import MultiSpectralAttentionLayer

class newdp(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(newdp, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False, groups=groups)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.sig = nn.Sigmoid()
    def forward(self, r, d, i):
        a = self.avg(i)
        b = self.max(i)
        s = self.sig(a * r + b * d)
        res = r * s + d * s
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
        print(f"RGB SwinTransformer loading pre_model ${pre_model1}")
        print(f"Depth SwinTransformer loading pre_model ${pre_model1}")

    def __init__(self, mode='small'):
        super(Me, self).__init__()
        self.resnet = shunted_b()
        self.resnet2 = shunted_b()

        self.config = Config()
        # self.ini = BasicConv2d(3, 256, 1)

        self.zc = BasicConv2d(12, 3, 1)

        # self.hfx2 = BasicConv2d(512, 128, 1)
        # self.hfx2ca = CM(128)
        # self.hfy2 = BasicConv2d(512, 128, 1)
        # self.hfy2ca = CM(128)

        self.col4x = BasicConv2d(128, 256, 1, 2)
        self.col4y = BasicConv2d(128, 256, 1, 2)

        # self.ca1 = SELayer(128)
        # self.ca2 = SELayer(128)
        # self.ca3 = SELayer(512)
        # self.ca4 = SELayer(512)

        # self.sa1 = SpatialAttention(7)
        # self.sa2 = SpatialAttention(7)
        # self.sa3 = SpatialAttention(7)
        # self.sa4 = SpatialAttention(7)
        #
        # self.sa1_2 = SpatialAttention(7)
        # self.sa2_2 = SpatialAttention(7)
        # self.sa3_2 = SpatialAttention(7)
        # self.sa4_2 = SpatialAttention(7)
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

        self.sa1_3 = MultiSpectralAttentionLayer(64, 64, 64)
        self.sa2_3 = MultiSpectralAttentionLayer(128, 32, 32)
        self.sa3_3 = MultiSpectralAttentionLayer(256, 16, 16)
        self.sa4_3 = MultiSpectralAttentionLayer(512, 8, 8)

        # self.sa1_3 = MultiSpectralAttentionLayer(64, 64, 64)
        # self.sa2_3 = MultiSpectralAttentionLayer(128, 32, 32)
        # self.sa3_3 = MultiSpectralAttentionLayer(320, 16, 16)
        # self.sa4_3 = MultiSpectralAttentionLayer(512, 8, 8)

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
        self.res3 = BasicConv2d(128, 1, 1)
        self.end = BasicConv2d(128, 1, 3, 1, 1)

        self.supself1 = BasicConv2d(64, 1, 1)

        # self.dct1 = MultiSpectralAttentionLayer(64, )
        self.aspp1 = ASPP(256, [7, 5, 3, 1], 256)
        self.aspp2 = ASPP(128, [7, 5, 3, 1], 128)
        self.aspp3 = ASPP(64, [7, 5, 3, 1], 64)

        self.newdp = newdp(512, 512, 3, 1, 1, 1, 1)
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
        patch_embed = getattr(self.resnet, f"patch_embed{1}")
        block = getattr(self.resnet, f"block{1}")
        norm = getattr(self.resnet, f"norm{1}")
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
        patch_embed = getattr(self.resnet, f"patch_embed{2}")
        block = getattr(self.resnet, f"block{2}")
        norm = getattr(self.resnet, f"norm{2}")
        y, H, W = patch_embed(y)
        for blk in block:
            y = blk(y, H, W)
        y = norm(y)
        y = y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        d.append(y)

        # ------ colomn3------
        x = x_he * x + x
        y = y_he * y + x

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
        patch_embed = getattr(self.resnet, f"patch_embed{3}")
        block = getattr(self.resnet, f"block{3}")
        norm = getattr(self.resnet, f"norm{3}")
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
        w.append(z)

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
        patch_embed = getattr(self.resnet, f"patch_embed{4}")
        block = getattr(self.resnet, f"block{4}")
        norm = getattr(self.resnet, f"norm{4}")
        y, H, W = patch_embed(y)
        for blk in block:
            y = blk(y, H, W)
        y = norm(y)
        y = y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        d.append(y)

        x = x * x_he + x
        r.append(x)
        y = y * y_he + x
        d.append(y)
        # x = self.ca3(x_he) + x
        # r.append(x)
        # y = self.ca4(y_he) + y
        # d.append(y)
        # for i in r:
        #     print("r",i.size())
        # for i in d:
        #     print("d",i.size())
        # for i in z:
        #     print("z",i.size())

        rd = []
        # re = self.sa1(r[0]) * d[0] + self.sa1_2(r[0]) * d[0]
        # rd.append(re + r[0])
        # re = self.sa2(r[1]) * d[1] + self.sa2_2(r[1]) * d[1]
        # rd.append(re + r[1])
        # re = self.sa3(r[2]) * d[2] + self.sa3_2(r[2]) * d[2]
        # rd.append(re + r[2])
        # re = self.sa4(r[3]) * d[3] + self.sa4_2(r[3]) * d[3]
        # rd.append(r[3] * re + r[3])
        re = self.sa1_3(self.re1(torch.cat((self.sa1(r[0]), self.sa1_2(d[0])), dim=1)))
        rd.append(re)
        re = self.sa2_3(self.re2(torch.cat((self.sa2(r[1]), self.sa2_2(d[1])), dim=1)))
        rd.append(re)
        re = self.sa3_3(self.re3(torch.cat((self.sa3(r[2]), self.sa3_2(d[2])), dim=1)))
        rd.append(re)
        re = self.sa4_3(self.re4(torch.cat((self.sa4(r[3]), self.sa4_2(d[3])), dim=1)))
        rd.append(re)
        # for i in rd:
        #     print("rd", i.size())
        # original
        # d0 = self.doc(r[-1] + d[-1])
        # now '
        # print(r[-1].size(), d[-1].size(), w[-1].size())
        d0 = self.newdp(r[-1], d[-1], w[-1])
        d0 = self.prs0(F.interpolate(d0, scale_factor=2))
        x = self.DD1(rd[-1], rd[-2])
        y = self.aspp1(d0)
        d1 = x * y + x + y

        d1 = self.prs1(F.interpolate(d1, scale_factor=2))
        x = self.DD2(rd[-1], rd[-3])
        y = self.aspp2(d1)
        d2 = x * y + x + y

        d2 = self.prs2(F.interpolate(d2, scale_factor=2))
        d3 = torch.cat((self.DD3(rd[-1], rd[-4]), d2), dim=1)
        x = self.DD3(rd[-1], rd[-4])
        y = self.aspp3(d2)
        d2 = x * y + x + y
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

        if self.training:
            return returnvalues, returnr, returnd
        else:
            return returnvalues


if __name__ == '__main__':
    a = torch.randn(2, 3, 256, 256).cuda()
    b = torch.randn(2, 3, 256, 256).cuda()
    model = Me()
    model.cuda()
    model.load_pre("/media/wby/shuju/ckpt_B.pth")
    out = model(a, b)
    # end = time.time()

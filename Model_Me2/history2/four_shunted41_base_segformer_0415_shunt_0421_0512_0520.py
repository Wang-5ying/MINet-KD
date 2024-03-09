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
from codes.GCoNet_plus_For_Four_Model.Model_Me.module import DD, SpatialAttention, CM, BasicConv2d, SELayer, ASPP, MixerBlock
# from backbone.Shunted_Transformer.SSA import shunted_b
# from mmseg.models.backbones.mix_transformer import mit_b5
from backbone.Shunted_Transformer.SSA import shunted_b
from 文献代码.FcaNet.model.layer import MultiSpectralAttentionLayer


class Me(nn.Module):
    def load_pre(self, pre_model1):
        new_state_dict3 = OrderedDict()
        state_dict = torch.load(pre_model1)
        for k, v in state_dict.items():
            name = k
            new_state_dict3[name] = v
        self.resnet.load_state_dict(new_state_dict3, strict=False)
        # self.resnet2.load_state_dict(new_state_dict3, strict=False)
        # self.resnet3.load_state_dict(new_state_dict3, strict=True)
        print(f"RGB SwinTransformer loading pre_model ${pre_model1}")
        print(f"Depth SwinTransformer loading pre_model ${pre_model1}")

    def __init__(self, mode='small'):
        super(Me, self).__init__()
        self.resnet = shunted_b()
        self.resnet2 = shunted_b()
        # self.resnet3 = shunted_b()

        self.config = Config()
        # self.ini = BasicConv2d(3, 256, 1)

        self.zc = BasicConv2d(12, 3, 1)

        self.col4x = BasicConv2d(128, 256, 1, 2)
        self.col4y = BasicConv2d(128, 256, 1, 2)


        self.sa1 = MultiSpectralAttentionLayer(64, 64, 64)
        self.sa2 = MultiSpectralAttentionLayer(128, 32, 32)
        self.sa3 = MultiSpectralAttentionLayer(256, 16, 16)
        self.sa4 = MultiSpectralAttentionLayer(512, 8, 8)

        self.re1 = nn.Conv2d(64 * 2, 64, 1)
        self.re2 = nn.Conv2d(128 * 2, 128, 1)
        self.re3 = nn.Conv2d(256 * 2, 256, 1)
        self.re4 = nn.Conv2d(512 * 2, 512, 1)

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

        self.mlp1 = MixerBlock(256, 128, 256) # 16 * 16
        self.mlp2 = MixerBlock(128, 64, 1024)
        self.mlp3 = MixerBlock(64, 32, 4096)
        self.mlp1_1  = MixerBlock(256, 128, 256)  # 16 * 16
        self.mlp2_1 = MixerBlock(128, 64, 1024)
        self.mlp3_1 = MixerBlock(64, 32, 4096)
        self.mlp_decode1 = DecoderHead([64, 128, 256, 512])

        self.red1 = nn.Conv2d(512, 256, 1)
        self.red2 = nn.Conv2d(256, 128, 1)
        self.red3 = nn.Conv2d(128, 64, 1)
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



        rd = []

        re = self.re1(torch.cat((self.sa1(r[0]), self.sa1(d[0])), dim=1))
        rd.append(re)
        re = self.re2(torch.cat((self.sa2(r[1]), self.sa2(d[1])), dim=1))
        rd.append(re)
        re = self.re3(torch.cat((self.sa3(r[2]), self.sa3(d[2])), dim=1))
        rd.append(re)
        re = self.re4(torch.cat((self.sa4(r[3]), self.sa4(d[3])), dim=1))
        rd.append(re)
        # for i in rd:
        #     print("rd", i.size())
        decode = []

        d0 = self.doc(r[-1] + d[-1])
        decode.append(d0)

        d0 = self.prs0(F.interpolate(d0, scale_factor=2)) #  256, 16, 16
        # print(d0.size())
        patch_embed = getattr(self.resnet2, f"patch_embed{4}")
        block = getattr(self.resnet2, f"block{4}")
        norm = getattr(self.resnet2, f"norm{4}")
        d0, H, W = patch_embed(d0)
        for blk in block:
            d0 = blk(d0, H, W)
        d0 = norm(d0)
        d0 = d0.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 512, 8, 8
        d0 = F.interpolate(self.red1(d0), scale_factor=2)
        # print(d0.size())
        x = self.DD1(rd[-1], rd[-2])
        # y = self.aspp1(d0)
        d1 = self.mlp1(x * d0 + x + d0)
        d1 = self.mlp1_1(d1)
        decode.append(d1)

        d1 = self.prs1(F.interpolate(d1, scale_factor=2))
        patch_embed = getattr(self.resnet2, f"patch_embed{3}")
        block = getattr(self.resnet2, f"block{3}")
        norm = getattr(self.resnet2, f"norm{3}")
        d1, H, W = patch_embed(d1)
        for blk in block:
            d1 = blk(d1, H, W)
        d1 = norm(d1)
        d1 = d1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        d1 = F.interpolate(self.red2(d1), scale_factor=2)
        x = self.DD2(rd[-1], rd[-3])
        # y = self.aspp2(d1)
        d2 = self.mlp2(x * d1 + x + d1)
        d2 = self.mlp2_1(d2)
        decode.append(d2)

        d2 = self.prs2(F.interpolate(d2, scale_factor=2))
        patch_embed = getattr(self.resnet2, f"patch_embed{2}")
        block = getattr(self.resnet2, f"block{2}")
        norm = getattr(self.resnet2, f"norm{2}")
        d2, H, W = patch_embed(d2)
        for blk in block:
            d2 = blk(d2, H, W)
        d2 = norm(d2)
        d2 = d2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        d2 = F.interpolate(self.red3(d2), scale_factor=2)
        x = self.DD3(rd[-1], rd[-4])
        # y = self.aspp3(d2)
        d3 = self.mlp3(x * d2 + x + d2)
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

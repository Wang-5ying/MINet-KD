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
from codes.GCoNet.models.GCoNet import AllAttLayer

class Me(nn.Module):
    def load_pre(self, pre_model1):
        new_state_dict3 = OrderedDict()
        state_dict = torch.load(pre_model1)
        for k, v in state_dict.items():
            name = k
            new_state_dict3[name] = v
        self.resnet.load_state_dict(new_state_dict3, strict=False)
        self.resnet2.load_state_dict(new_state_dict3, strict=False)
        self.resnet2.load_state_dict(new_state_dict3, strict=True)
        print(f"RGB SwinTransformer loading pre_model ${pre_model1}")
        print(f"Depth SwinTransformer loading pre_model ${pre_model1}")

    def __init__(self, mode='small'):
        super(Me, self).__init__()
        self.resnet = shunted_b()
        self.resnet2 = shunted_b()
        self.resnet3 = shunted_b()

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
        self.re4 = torch.nn.ConvTranspose2d(512 * 2, 512, 3, 2, 1, 1, 1, True, 1)

        self.sa1_2 = MultiSpectralAttentionLayer(64, 64, 64)
        self.sa2_2 = MultiSpectralAttentionLayer(128, 32, 32)
        self.sa3_2 = MultiSpectralAttentionLayer(256, 16, 16)
        self.sa4_2 = MultiSpectralAttentionLayer(512, 8, 8)

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
        self.mlp1_1  = MixerBlock(256, 128, 256)  # 16 * 16
        self.mlp2_1 = MixerBlock(128, 64, 1024)
        self.mlp3_1 = MixerBlock(64, 32, 4096)
        self.mlp_decode1 = DecoderHead([64, 128, 256, 512])

        self.coa0 = AllAttLayer(512)
        self.coa1 = AllAttLayer(320)
        self.coa2 = AllAttLayer(128)

        self.precoa1 = torch.nn.ConvTranspose2d(512, 320, 3, 2, 1, 1, 1, True, 1)
        self.precoa2 = torch.nn.ConvTranspose2d(320, 128, 3, 2, 1, 1, 1, True, 1)
        # self.precoa3 = torch.nn.ConvTranspose2d(128, 64, 3, 2, 1, 1, 1, True, 1)
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

        x = x * x_he + x
        r.append(x)
        y = y * y_he + x
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
        decode = []

        coa1, d0 = self.coa0(rd[-1])
        decode.append(d0)

        d0 = self.prs0(d0)
        x = self.DD1(rd[-1], rd[-2] * coa1)
        y = self.aspp1(d0)
        d1 = self.mlp1(x * y + x + y)
        d1 = self.mlp1_1(d1)
        decode.append(d1)

        # coa2, ca2 = self.coa1(self.precoa1(rd[-1]))
        coa2 = F.interpolate(coa1, scale_factor=2)
        d1 = self.prs1(F.interpolate(d1, scale_factor=2))
        x = self.DD2(rd[-1], rd[-3] * coa2)
        y = self.aspp2(d1)
        d2 = self.mlp2(x * y + x + y)
        d2 = self.mlp2_1(d2)
        decode.append(d2)

        # coa3, _ = self.coa2(self.precoa2(ca2))
        coa3 = F.interpolate(coa2, scale_factor=2)
        d2 = self.prs2(F.interpolate(d2, scale_factor=2))
        x = self.DD3(rd[-1], rd[-4] * coa3)
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

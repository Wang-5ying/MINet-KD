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
from codes.GCoNet_plus_For_Four_Model.Model_Me.module import DD, SpatialAttention, CM, BasicConv2d, SELayer, ASPP, MixerBlock, SpatialAttention, CA
# from backbone.Shunted_Transformer.SSA import shunted_b
# from mmseg.models.backbones.mix_transformer import mit_b4
from backbone.Shunted_Transformer.SSA import shunted_b
from 文献代码.FcaNet.model.layer import MultiSpectralAttentionLayer
from codes.GCoNet.models.GCoNet import AllAttLayer
from backbone.VanillaNet.models.vanillanet import vanillanet_10
class Me(nn.Module):
    def load_pre(self, pre_model1):
        new_state_dict3 = OrderedDict()
        state_dict = torch.load(pre_model1)["model"]
        for k, v in state_dict.items():
            name = k
            new_state_dict3[name] = v
        self.resnet.load_state_dict(new_state_dict3, strict=True)
        self.resnet2.load_state_dict(new_state_dict3, strict=True)
        self.resnet3.load_state_dict(new_state_dict3, strict=True)
        print(f"RGB SwinTransformer loading pre_model ${pre_model1}")
        print(f"Depth SwinTransformer loading pre_model ${pre_model1}")

    def __init__(self, mode='small'):
        super(Me, self).__init__()
        # self.resnet = shunted_b()
        # self.resnet2 = shunted_b()
        # self.resnet3 = shunted_b()
        self.resnet = vanillanet_10()
        self.resnet2 = vanillanet_10()
        self.resnet3 = vanillanet_10()
        self.config = Config()
        # self.ini = BasicConv2d(3, 256, 1)

        self.zc = BasicConv2d(12, 3, 1)

        self.cca1 = CA(512)
        self.cca2 = CA(512)
        self.csa1 = SpatialAttention(3)
        self.csa2 = SpatialAttention(3)
        self.cca3 = CA(2048)
        self.cca4 = CA(2048)
        self.csa3 = SpatialAttention(3)
        self.csa4 = SpatialAttention(3)

        self.sa1 = MultiSpectralAttentionLayer(512, 64, 64)
        self.sa2 = MultiSpectralAttentionLayer(1024, 32, 32)
        self.sa3 = MultiSpectralAttentionLayer(2048, 16, 16)
        self.sa4 = MultiSpectralAttentionLayer(4096, 8, 8)

        self.re1 = nn.Conv2d(512 * 2, 64, 1)
        self.re2 = nn.Conv2d(1024 * 2, 128, 1)
        self.re3 = nn.Conv2d(2048 * 2, 256, 1)
        self.re4 = nn.Conv2d(4096 * 2, 512, 1)

        self.sa1_2 = MultiSpectralAttentionLayer(512, 64, 64)
        self.sa2_2 = MultiSpectralAttentionLayer(1024, 32, 32)
        self.sa3_2 = MultiSpectralAttentionLayer(2048, 16, 16)
        self.sa4_2 = MultiSpectralAttentionLayer(4096, 8, 8)

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
        # self.supself1 = BasicConv2d(512, 1, 1)

        # self.dct1 = MultiSpectralAttentionLayer(64, )
        self.aspp1 = ASPP(256, [7, 5, 3, 1], 256)
        self.aspp2 = ASPP(128, [7, 5, 3, 1], 128)
        self.aspp3 = ASPP(64, [7, 5, 3, 1], 64)

        self.mlp1 = MixerBlock(256, 128, 256)  # 16 * 16
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


    def forward(self, x, y):
        r = []
        d = []
        w = []
        B = x.shape[0]
        z = self.zc(torch.cat((x, y, x - y, x + y), dim=1))
        # ----------colomn 1--------------
        # r_stage1
        x = self.resnet.stem1(x)
        x = torch.nn.functional.leaky_relu(x, self.resnet.act_learn)
        x = self.resnet.stem2(x)
        x = self.resnet.stages[0](x)
        r.append(x)

        # d_stage1
        y = self.resnet3.stem1(y)
        y = torch.nn.functional.leaky_relu(y, self.resnet3.act_learn)
        y = self.resnet3.stem2(y)
        y = self.resnet3.stages[0](y)
        d.append(y)

        # z_stage1
        z = self.resnet2.stem1(z)
        z = torch.nn.functional.leaky_relu(z, self.resnet2.act_learn)
        z = self.resnet2.stem2(z)
        z = self.resnet2.stages[0](z)
        w.append(z)

        # ------- colomn2 ---------
        # z_stage2
        x = self.csa1(self.cca1(x + z)) + x
        y = self.csa2(self.cca2(y + z)) + y
        z = x + y + z
        z = torch.cat((x, y, z), dim=0)
        z = self.resnet2.stages[1](z)
        b, c, h, w1 = z.size()
        x_he = z[0:int(b / 3), :, :, :]
        y_he = z[int(b / 3):2 * int(b / 3), :, :]
        z = z[2 * int(b / 3):b, :, :, :]

        # r_stage2
        x = self.resnet.stages[1](x)
        r.append(x)

        # d_stage2
        y = self.resnet3.stages[1](y)
        d.append(y)

        # ------ colomn3------
        # x = x_he * x + x
        # y = y_he * y + x
        x = self.soft1(self.gap1(x_he)) * x + x
        y = self.soft2(self.gap2(y_he)) * y + y
        # r_stage3
        x = self.resnet.stages[2](x)
        x = self.resnet.stages[3](x)
        x = self.resnet.stages[4](x)
        x = self.resnet.stages[5](x)
        r.append(x)

        # d_stage3
        y = self.resnet3.stages[2](y)
        y = self.resnet3.stages[3](y)
        y = self.resnet3.stages[4](y)
        y = self.resnet3.stages[5](y)
        d.append(y)

        # z_stage3
        z = self.resnet2.stages[2](z)
        z = self.resnet2.stages[3](z)
        z = self.resnet2.stages[4](z)
        z = self.resnet2.stages[5](z)
        w.append(z)

        # ------- colomn4 ---------
        # z_stage4
        x = self.csa3(self.cca3(x + z)) + x
        y = self.csa4(self.cca4(y + z)) + y
        z = x + y + z
        z = torch.cat((x, y, z), dim=0)
        b, c, h, w1 = z.size()
        # z = self.zen1(z)
        z = self.resnet2.stages[6](z)
        z = self.resnet2.stages[7](z)
        b, c, h, w1 = z.size()
        x_he = z[0:int(b / 3), :, :, :]
        y_he = z[int(b / 3):2 * int(b / 3), :, :]
        z = z[2 * int(b / 3):b, :, :, :]

        # r_stage4
        x = self.resnet.stages[6](x)
        x = self.resnet.stages[7](x)
        r.append(x)

        # d_stage4
        y = self.resnet3.stages[6](y)
        y = self.resnet3.stages[7](y)
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
        x = self.DD3(rd[-1], rd[-4])
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
    # model.load_pre("/media/wby/shuju/ckpt_B.pth")
    out = model(a, b)
    # end = time.time()

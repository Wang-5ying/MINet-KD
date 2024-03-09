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
from codes.GCoNet_plus_For_Four_Model.Model_Me.module import DD, SpatialAttention, CM, BasicConv2d, SELayer, ASPP, MixerBlock, SpatialAttention, CA, MAttention
# from backbone.Shunted_Transformer.SSA import shunted_b
# from mmseg.models.backbones.mix_transformer import mit_b4
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

        self.td1 = torch.nn.ConvTranspose2d(512, 256, 3, 2, 1, 1, 1, True, 1)
        self.td2 = torch.nn.ConvTranspose2d(256, 128, 3, 2, 1, 1, 1, True, 1)
        self.td3 = torch.nn.ConvTranspose2d(128, 64, 3, 2, 1, 1, 1, True, 1)
        self.td4 = torch.nn.ConvTranspose2d(64, 32, 3, 2, 1, 1, 1, True, 1)
        self.td5 = torch.nn.ConvTranspose2d(32, 3, 3, 2, 1, 1, 1, True, 1)


        self.sup1 = BasicConv2d(256, 1, 3, 1, 1)
        self.sup2 = BasicConv2d(128, 1, 3, 1, 1)
        self.sup3 = BasicConv2d(64, 1, 3, 1, 1)
        self.sup4 = BasicConv2d(3, 1, 3, 1, 1)

        self.glo = MAttention(512)
        self.cls = nn.Conv2d(256, 1, 1)
        self.adpgl1 = torch.nn.ConvTranspose2d(512, 256, 3, 2, 1, 1, 1, True, 1)
        self.adpgl2 = torch.nn.ConvTranspose2d(256, 128, 3, 2, 1, 1, 1, True, 1)
        self.adpgl3 = torch.nn.ConvTranspose2d(128, 64, 3, 2, 1, 1, 1, True, 1)
        self.adpgl4 = torch.nn.ConvTranspose2d(64, 32, 3, 2, 1, 1, 1, True, 1)
        self.adpgl5 = torch.nn.ConvTranspose2d(32, 3, 3, 2, 1, 1, 1, True, 1)

        self.adp1avg = nn.AdaptiveAvgPool2d(32)
        self.sig = nn.Sigmoid()
    def forward(self, x, y):
        r = []
        d = []
        w = []
        r = self.resnet.forward_features(x)
        d = self.resnet.forward_features(y)
        B = x.shape[0]
        rd = []

        re = self.re1(torch.cat((self.sa1(r[0]), self.sa1_2(d[0])), dim=1))
        rd.append(re)
        re = self.re2(torch.cat((self.sa2(r[1]), self.sa2_2(d[1])), dim=1))
        rd.append(re)
        re = self.re3(torch.cat((self.sa3(r[2]), self.sa3_2(d[2])), dim=1))
        rd.append(re)
        re = self.re4(torch.cat((self.sa4(r[3]), self.sa4_2(d[3])), dim=1))
        rd.append(re)

        decode = []
        global1 = self.glo(rd[-1])

        res0 = self.doc(r[-1] + d[-1])
        decode.append(res0)

        globali = []
        res1 = self.td1(res0) + rd[-2]
        global1 = self.adpgl1(global1)
        globali.append(global1)
        cls = self.sig(self.adp1avg(self.cls(global1)))
        res1 = global1 * res1 + res1
        # print("glo", global1.size(), res1.size())
        patch_embed = getattr(self.resnet3, f"patch_embed{4}")
        block = getattr(self.resnet3, f"block{4}")
        norm = getattr(self.resnet3, f"norm{4}")
        res1, H, W = patch_embed(res1)
        for blk in block:
            res1 = blk(res1, H, W)
        res1 = norm(res1)
        res1 = res1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        res1 = self.td1(res1)
        decode.append(res1)

        res2 = self.td2(res1) + rd[-3]
        global2 = self.adpgl2(global1)
        globali.append(global2)
        res2 = global2 * res2 + res2
        patch_embed = getattr(self.resnet3, f"patch_embed{3}")
        block = getattr(self.resnet3, f"block{3}")
        norm = getattr(self.resnet3, f"norm{3}")
        res2, H, W = patch_embed(res2)
        for blk in block:
            res2 = blk(res2, H, W)
        res2 = norm(res2)
        res2 = res2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        res2 = self.td2(res2)
        decode.append(res2)

        res3 = self.td3(res2) + rd[-4]
        global3 = self.adpgl3(global2)
        globali.append(global3)
        res3 = global3 * res3 + res3
        patch_embed = getattr(self.resnet3, f"patch_embed{2}")
        block = getattr(self.resnet3, f"block{2}")
        norm = getattr(self.resnet3, f"norm{2}")
        res3, H, W = patch_embed(res3)
        for blk in block:
            res3 = blk(res3, H, W)
        res3 = norm(res3)
        res3 = res3.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        res3 = self.td3(res3)
        decode.append(res3)

        res4 = self.td5(self.td4(res3))
        global4 = self.adpgl5(self.adpgl4(global3))
        globali.append(global4)
        res4 = global4 * res4 + res4
        patch_embed = getattr(self.resnet3, f"patch_embed{1}")
        block = getattr(self.resnet3, f"block{1}")
        norm = getattr(self.resnet3, f"norm{1}")
        res4, H, W = patch_embed(res4)
        for blk in block:
            res4 = blk(res4, H, W)
        res4 = norm(res4)
        res4 = res4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        res4 = self.td5(self.td4(res4))
        decode.append(res4)

        res1 = F.interpolate(self.sup1(res1), 256)
        res2 = F.interpolate(self.sup2(res2), 256)
        res3 = F.interpolate(self.sup3(res3), 256)
        res4 = F.interpolate(self.sup4(res4), 256)

        max_value, max_index = torch.max(res4, dim=1)
        B, H, W = max_value.size()
        max_value = max_value.view(B, H * W)
        relation = max_value @ max_value.transpose(0, 1)
        # print(res0.size(), res1.size(), res2.size(), res3.size())
        # decoder = [d0, d0, d0, d0]
        # decoder[0] = decode[-1]
        # decoder[1] = decode[-2]
        # decoder[2] = decode[-3]
        # decoder[3] = decode[-4]
        #
        # # mlp_decode = F.interpolate(self.mlp_decode1(decoder), size=256)
        # # print(d3.size(),"d3")
        # res0 = F.interpolate(self.res0(d0), size=256)
        # res1 = F.interpolate(self.res1(d1), size=256)
        # res2 = F.interpolate(self.res2(d2), size=256)
        # res3 = F.interpolate(self.res3(d3), size=256)
        # res = F.interpolate(self.end(d3), size=256)
        returnvalues = []
        # returnvalues.append(res0)
        returnvalues.append(res1)
        returnvalues.append(res2)
        returnvalues.append(res3)
        returnvalues.append(res4)
        # returnvalues.append(mlp_decode)

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
        # for i in w:
        #     print("r", i.size())

        if self.training:
            return returnvalues, r, d
        else:
            return returnvalues
        # else:
        #     return returnvalues
        # return returnvalues, r, d, cls


if __name__ == '__main__':
    a = torch.randn(2, 3, 256, 256).cuda()
    b = torch.randn(2, 3, 256, 256).cuda()
    model = Me()
    model.cuda()
    # model.load_pre("/media/wby/shuju/ckpt_B.pth")
    out = model(a, b)
    # end = time.time()

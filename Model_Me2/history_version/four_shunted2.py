import importlib
from math import sqrt
import time
import numpy as np
import torch
from mmseg.models import build_segmentor
from torch import nn
import torch.nn.functional as F
# from MLNet.ResT.models.rest_v2 import ResTV2
from codes.bayibest82.baseapi.newapii711715 import BasicConv2d, node, CA, GM2  # atttention
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
from codes.GCoNet_plus_For_Four_Model.Model_Me.module import DD, SpatialAttention
from backbone.Shunted_Transformer.SSA import shunted_t
class Me(nn.Module):
    def load_pre(self, pre_model1):
        new_state_dict3 = OrderedDict()
        state_dict = torch.load(pre_model1)
        # for k, v in state_dict.items():
        #     name = k[9:]
        #     new_state_dict3[name] = v
        # self.decoder.load_state_dict(new_state_dict3, strict=False)
        self.resnet.load_state_dict(state_dict, strict=True)
        print(f"RGB SwinTransformer loading pre_model ${pre_model1}")
        print(f"Depth SwinTransformer loading pre_model ${pre_model1}")

    def __init__(self, mode='small'):
        super(Me, self).__init__()
        self.resnet = shunted_t()
        self.resnet2 = shunted_t()
        self.config = Config()
        # self.ini = BasicConv2d(3, 256, 1)

        self.bas = BasicConv2d(64, 64, 3, 1, 1)
        self.end = BasicConv2d(64, 1, 3, 1, 1)

        self.DD1 = DD(512, 256, 4, 3, 2)
        self.DD2 = DD(512, 128, 8, 7, 2)
        self.DD3 = DD(512, 64, 16, 15, 2)

        self.sa1 = SpatialAttention(7)
        self.sa2 = SpatialAttention(7)
        self.sa3 = SpatialAttention(7)
        self.sa4 = SpatialAttention(7)

        self.res1 = BasicConv2d(128, 1, 1)
        self.res2 = BasicConv2d(64, 1, 1)
        self.res3 = BasicConv2d(64, 1, 1)

        self.prs1 = BasicConv2d(256, 128, 1)
        self.prs2 = BasicConv2d(128, 64, 1)
    def forward(self, x, y):
        r = []
        B = x.shape[0]

        for i in range(self.resnet.num_stages):
            patch_embed = getattr(self.resnet, f"patch_embed{i + 1}")
            block = getattr(self.resnet, f"block{i + 1}")
            norm = getattr(self.resnet, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            if i != self.resnet.num_stages:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                r.append(x)
        d = []
        B = y.shape[0]

        for i in range(self.resnet2.num_stages):
            patch_embed = getattr(self.resnet2, f"patch_embed{i + 1}")
            block = getattr(self.resnet2, f"block{i + 1}")
            norm = getattr(self.resnet2, f"norm{i + 1}")
            y, H, W = patch_embed(y)
            for blk in block:
                y = blk(y, H, W)
            y = norm(y)
            if i != self.resnet2.num_stages:
                y = y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                d.append(y)

        rd = []
        re = self.sa1(d[0]) * d[0]
        rd.append(r[0] * re + r[0])
        re = self.sa2(d[1]) * d[1]
        rd.append(r[1] * re + r[1])
        re = self.sa3(d[2]) * d[2]
        rd.append(r[2] * re + r[2])
        re = self.sa4(d[3]) * d[3]
        rd.append(r[3] * re + r[3])

        # for i in rd:
        #     print(i.size())
        d1 = self.DD1(rd[-1], rd[-2])
        d1 = self.prs1(F.interpolate(d1, scale_factor=2))
        d2 = self.DD2(rd[-1], rd[-3]) + d1
        d2 = self.prs2(F.interpolate(d2, scale_factor=2))
        d3 = self.DD3(rd[-1], rd[-4]) + d2
        # print(d3.size(),"d3")
        res1 = F.interpolate(self.res1(d1), size=256)
        res2 = F.interpolate(self.res2(d2), size=256)
        res3 = F.interpolate(self.res3(d3), size=256)
        res = F.interpolate(self.end(d3), size=256)

        return res1, res2, res3, res


if __name__ == '__main__':
    a = torch.randn(2, 3, 224, 224).cuda()
    b = torch.randn(2, 3, 224, 224).cuda()
    model = Me()
    model.cuda()
    model.load_pre("/media/wby/shuju/ckpt_T.pth")
    out = model(a, b)
    # end = time.time()

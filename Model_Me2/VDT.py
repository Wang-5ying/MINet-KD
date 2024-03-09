from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
# from VDT.LUCKY.swinNet import SwinTransformer
from LUCKY.api import RRAA, End, BasicConv2d, de, edge, AGG2
# from third.mix_transformer import mit_b5
from mmseg.models import build_segmentor
from timm.models import create_model
from mmcv import Config
from third.mix_transformer import mit_b5
from torchvision.models import resnet50
from plug_and_play_modules.DO_Conv.do_conv_pytorch_1_10 import DOConv2d


class CA(nn.Module):
    def __init__(self, in_ch):
        super(CA, self).__init__()
        self.avg_weight = nn.AdaptiveAvgPool2d(1)
        self.max_weight = nn.AdaptiveMaxPool2d(1)
        self.fus = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(in_ch // 2, in_ch, 1, 1, 0),
        )
        self.c_mask = nn.Sigmoid()

    def forward(self, x):
        avg_map_c = self.avg_weight(x)
        max_map_c = self.max_weight(x)
        c_mask = self.c_mask(torch.add(self.fus(avg_map_c), self.fus(max_map_c)))
        return torch.mul(x, c_mask)


class SA(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, ):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Do_convrlu(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, ):
        super(Do_convrlu, self).__init__()
        self.conv = nn.Sequential(
            DOConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding))
        # nn.ReLU(),
        # DOConv2d(out_planes, out_planes,  kernel_size=kernel_size, stride=1, padding=padding),
        # nn.ReLU())

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        return x


class Bottlenceck2D(nn.Module):
    def __init__(self, in_planes, planes, stride=1, padding=0, dilation=1):
        super(Bottlenceck2D, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, in_planes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out = out + residual
        return out


class Bottlenceck2D_Do(nn.Module):
    def __init__(self, in_planes, planes, stride=1, padding=0, dilation=1):
        super(Bottlenceck2D_Do, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = DOConv2d(in_planes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = DOConv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = DOConv2d(planes, in_planes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out = out + residual
        return out


class Fusion(nn.Module):
    def __init__(self, in_planes, in_planes2, inplanes3, size):
        super(Fusion, self).__init__()
        print(in_planes)
        self.doconv = DOConv2d(in_planes, 64, 1)
        self.doconv2 = DOConv2d(in_planes2, 64, 1)
        self.size = size
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU()
        )
        self.doconv2_ = DOConv2d(64, 64, 1)
        self.ca = CA(64)
        self.sa = SA(kernel_size=7)
        self.res = nn.Conv2d(inplanes3, 64, 1)
    def forward(self, up, down, res, glo, place):
        glo = F.interpolate(glo, self.size, mode='bilinear', align_corners=True)
        up = self.doconv(F.interpolate(up, self.size))
        down = self.doconv(F.interpolate(down, self.size))
        place = self.doconv2(F.interpolate(place, self.size))
        res = self.res(F.interpolate(res, self.size))

        glo = self.ca(glo) * glo
        glo = glo.unsqueeze(2)
        up = up.unsqueeze(2)
        down = down.unsqueeze(2)
        place = place.unsqueeze(2)
        udg = torch.cat((up, down, glo, place), dim=2)
        udg = self.conv3(self.conv3(udg))
        udg_1 = udg[:, :, 0:1, :, :]
        udg_2 = udg[:, :, 1:2, :, :]
        udg_3 = udg[:, :, 2:3, :, :]
        udg_4 = udg[:, :, 3:4, :, :]
        udg = udg_1 + udg_2 + udg_3 + udg_4
        udg = udg.squeeze(2)
        upg = self.doconv2_(udg) + udg

        weight = self.sa(udg)
        # print(weight.size(),res.size())
        res = F.interpolate(res, self.size) * weight + res

        res = res + upg
        return res
class TD(nn.Module):
    def __init__(self, in_planes):
        super(TD, self).__init__()
        self.con30 = nn.Sequential(
            nn.Conv3d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU()
        )
        self.sun4 = nn.Conv2d(in_planes * 3, 64, 3, 1, 1)

    def forward(self, r, t, d):
        r03d = r.unsqueeze(2)
        t03d = t.unsqueeze(2)
        d03d = d.unsqueeze(2)
        rtd0 = torch.cat((r03d, t03d, d03d), dim=2)
        rd1 = self.con30(self.con30(rtd0))
        rd1_1 = rd1[:, :, 0:1, :, :].squeeze(2)
        rd1_2 = rd1[:, :, 1:2, :, :].squeeze(2)
        rd1_3 = rd1[:, :, 2:3, :, :].squeeze(2)
        # rd1 = rd1_1 + rd1_2 + rd1_3
        # rd1 = rd1.squeeze(2)

        rd1 = self.sun4(torch.cat((rd1_1, rd1_2, rd1_3), dim=1))
        return rd1

class SRAA(nn.Module):
    def load_pre(self, pre_model):
        # r
        new_state_dict2 = OrderedDict()
        state_dict = torch.load("/media/user/shuju/segformer.b5.640x640.ade.160k.pth")['state_dict']
        for k, v in state_dict.items():
            name = k[9:]
            new_state_dict2[name] = v
        self.resnet.load_state_dict(new_state_dict2, False)
        # self.resnet.load_state_dict(torch.load(pre_model)['state_dict'], strict=True)
        print(f"RGB SwinTransformer loading pre_model ${pre_model}")
        # d
        new_state_dict2 = OrderedDict()
        state_dict = torch.load("/media/user/shuju/segformer.b5.640x640.ade.160k.pth")['state_dict']
        for k, v in state_dict.items():
            name = k[9:]
            new_state_dict2[name] = v
        self.resnet_d.load_state_dict(new_state_dict2, False)
        # self.resnet.load_state_dict(torch.load(pre_model)['state_dict'], strict=True)
        print(f"RGB SwinTransformer loading pre_model ${pre_model}")
        # t
        new_state_dict3 = OrderedDict()
        state_dict = torch.load("/media/user/shuju/segformer.b5.640x640.ade.160k.pth")['state_dict']
        for k, v in state_dict.items():
            name = k[9:]
            new_state_dict3[name] = v
        self.resnet_t.load_state_dict(new_state_dict3, False)
        # self.resnet.load_state_dict(torch.load(pre_model)['state_dict'], strict=True)
        print(f"RGB SwinTransformer loading pre_model ${pre_model}")

    def __init__(self):
        super(SRAA, self).__init__()
        self.resnet = mit_b5()
        self.resnet_d = mit_b5()
        self.resnet_t = mit_b5()
        self.guide = resnet50(pretrained=True)
        # he  2
        self.ca1 = CA(256)
        self.ca2 = CA(64)
        self.sum1 = Do_convrlu(256 + 64 * 2, 256, 1)
        self.split1 = Do_convrlu(512, 64, 1)

        # he  4
        self.ca1_2 = CA(1024)
        self.ca2_2 = CA(128)
        self.sum1_2 = Do_convrlu(1024 + 128 * 2, 1024, 1)
        self.split1_2 = Do_convrlu(2048, 128, 1)

        # he  6
        self.ca1_3 = CA(64)
        self.ca2_3 = CA(320)
        self.sum1_3 = Do_convrlu(64 + 320 * 2, 64, 1)
        self.split1_3 = Do_convrlu(128, 320, 1)

        # he  8
        self.ca1_4 = CA(320)
        self.ca2_4 = CA(512)
        self.sum1_4 = Do_convrlu(320 + 512 * 2, 320, 1)

        self.glo = Do_convrlu(512*3, 64, 1)
        self.gl = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU()
        )
        self.end1 = End(in_channel=64)

        self.sup1 = Do_convrlu(64, 1, 3, 1, 1)
        self.sup2 = Do_convrlu(64, 1, 3, 1, 1)
        self.sup3 = Do_convrlu(64, 1, 3, 1, 1)

        self.b1 = Do_convrlu(64, 64, 5, 1, 6, 3)
        self.b2 = Do_convrlu(64, 64, 5, 1, 6, 3)
        self.b3 = Do_convrlu(64, 64, 5, 1, 6, 3)
        self.b4 = Do_convrlu(64, 64, 5, 1, 6, 3)
        self.b5 = Do_convrlu(64, 64, 5, 1, 6, 3)
        # )

        self.d1 = de(1536, 512)
        self.d2 = de(1792, 256)
        self.d3 = de(1920, 128)

        self.edge1 = edge(64, 1)
        self.edge2 = edge(320, 1)

        self.xiaogl = Do_convrlu(1024, 64, 1)

        self.d = nn.Conv2d(1, 3, 1)

        self.e = nn.Conv2d(64, 1, 1)

        # 3D conv
        self.TD1 = TD(64)
        self.TD2 = TD(128)
        self.TD3 = TD(320)
        self.TD4 = TD(512)

        self.TD1_2 = TD(64)
        self.TD2_2 = TD(128)
        self.TD3_2 = TD(320)
        self.TD4_2 = TD(512)

        self.decbr1 = BasicConv2d(256, 64, 1)
        self.decbr2 = BasicConv2d(512, 128, 1)
        self.decbr3 = BasicConv2d(1024, 320, 1)
        self.decbr4 = BasicConv2d(2048, 512, 1)

        self.fusion1 = Fusion(64, 64, 64, 14)
        self.fusion2 = Fusion(64, 64, 64, 28)
        self.fusion3 = Fusion(64, 64, 64, 56)
        self.fusion4 = Fusion(64, 64, 64, 112)

        self.sup1 = nn.Conv2d(64, 1, 1)
        self.sup2 = nn.Conv2d(64, 1, 1)
        self.sup3 = nn.Conv2d(64, 1, 1)



    def forward(self, r, d, t):
        # encoder
        # stage 1
        # r
        brlayer_features = []
        B = r.shape[0]
        x = self.guide.conv1(r)
        x = self.guide.bn1(x)
        x = self.guide.relu(x)
        x = self.guide.maxpool(x)
        x = self.guide.layer1(x)
        brlayer_features.append(x)

        # d
        llayer_features = []
        B = d.shape[0]
        y, H, W = self.resnet_d.patch_embed1(d)
        for i, blk in enumerate(self.resnet_d.block1):
            y = blk(y, H, W)
        y = self.resnet_d.norm1(y)
        y = y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        llayer_features.append(y)

        # t
        t = self.d(t)
        B = t.shape[0]
        tlayer_features = []
        z, H, W = self.resnet_t.patch_embed1(t)
        for i, blk in enumerate(self.resnet_t.block1):
            z = blk(z, H, W)
        z = self.resnet_t.norm1(z)
        z = z.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        tlayer_features.append(z)

        # 和   2
        sum = self.sum1(
            torch.cat((self.ca1(brlayer_features[-1]), self.ca2(tlayer_features[-1]), self.ca2(llayer_features[-1])),
                      dim=1))
        x = self.guide.layer2(sum)
        brlayer_features.append(x)
        z = F.interpolate(self.split1(x), scale_factor=2)
        y = F.interpolate(self.split1(x), scale_factor=2)

        # stage  3
        # r
        x = self.guide.layer3(x)
        brlayer_features.append(x)
        # d
        y, H, W = self.resnet_d.patch_embed2(y)
        for i, blk in enumerate(self.resnet_d.block2):
            y = blk(y, H, W)
        y = self.resnet_d.norm2(y)
        y = y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        llayer_features.append(y)
        # t
        z, H, W = self.resnet_t.patch_embed2(z)
        for i, blk in enumerate(self.resnet_t.block2):
            z = blk(z, H, W)
        z = self.resnet_t.norm2(z)
        z = z.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        tlayer_features.append(z)
        # 和    4
        sum = self.sum1_2(
            torch.cat((F.interpolate(self.ca1_2(brlayer_features[-1]), size=48), self.ca2_2(tlayer_features[-1]),
                       self.ca2_2(llayer_features[-1])),
                      dim=1))
        x = self.guide.layer4(sum)
        brlayer_features.append(x)
        z = F.interpolate(self.split1_2(x), scale_factor=2)
        y = F.interpolate(self.split1_2(x), scale_factor=2)

        # 5
        rlayer_features = []
        x, H, W = self.resnet.patch_embed1(r)
        for i, blk in enumerate(self.resnet.block1):
            x = blk(x, H, W)
        x = self.resnet.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        rlayer_features.append(x)

        y, H, W = self.resnet_d.patch_embed3(y)
        for i, blk in enumerate(self.resnet_d.block3):
            y = blk(y, H, W)
        y = self.resnet_d.norm3(y)
        y = y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        llayer_features.append(y)
        # t
        z, H, W = self.resnet_t.patch_embed3(z)
        for i, blk in enumerate(self.resnet_t.block3):
            z = blk(z, H, W)
        z = self.resnet_t.norm3(z)
        z = z.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        tlayer_features.append(z)

        # 和    6
        sum = self.sum1_3(
            torch.cat((self.ca1_3(rlayer_features[-1]), F.interpolate(self.ca2_3(tlayer_features[-1]), size=96),
                       F.interpolate(self.ca2_3(llayer_features[-1]), size=96)),
                      dim=1))
        x, H, W = self.resnet.patch_embed2(sum)
        for i, blk in enumerate(self.resnet.block2):
            x = blk(x, H, W)
        x = self.resnet.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        rlayer_features.append(x)
        z = F.interpolate(self.split1_3(x), scale_factor=1 / 2)
        y = F.interpolate(self.split1_3(x), scale_factor=1 / 2)

        # 7
        # r
        x, H, W = self.resnet.patch_embed3(x)
        for i, blk in enumerate(self.resnet.block3):
            x = blk(x, H, W)
        x = self.resnet.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        rlayer_features.append(x)
        # d
        y, H, W = self.resnet_d.patch_embed4(y)
        for i, blk in enumerate(self.resnet_d.block4):
            y = blk(y, H, W)
        y = self.resnet_d.norm4(y)
        y = y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        llayer_features.append(y)
        # t
        z, H, W = self.resnet_t.patch_embed4(z)
        for i, blk in enumerate(self.resnet_t.block4):
            z = blk(z, H, W)
        z = self.resnet_t.norm4(z)
        z = z.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        tlayer_features.append(z)

        # 和    8
        sum = self.sum1_4(
            torch.cat((self.ca1_4(rlayer_features[-1]), F.interpolate(self.ca2_4(tlayer_features[-1]), size=24),
                       F.interpolate(self.ca2_4(llayer_features[-1]), size=24)),
                      dim=1))
        x, H, W = self.resnet.patch_embed4(sum)
        for i, blk in enumerate(self.resnet.block4):
            x = blk(x, H, W)
        x = self.resnet.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        rlayer_features.append(x)

        # print("rlayer")
        # for i in brlayer_features:
        #     print(i.size())
        # print("dlayer")
        # for i in llayer_features:
        #     print(i.size())
        #
        # print("tlayer")
        # for i in tlayer_features:
        #     print(i.size())
        # torch.Size([2, 64, 96, 96])
        # torch.Size([2, 128, 48, 48])
        # torch.Size([2, 320, 24, 24])
        # torch.Size([2, 512, 12, 12])
        # x1 = self.sa1(tlayer_features[0]) * llayer_features[0]
        # x2 = self.aca1(tlayer_features[0]) + x1
        # x3 = self.sa1(x2) * rlayer_features[0]
        # x4 = self.aca1(x2) * x1 + x3

        # decoder  上行分支
        # r03d = rlayer_features[0].unsqueeze(2)
        # t03d = tlayer_features[0].unsqueeze(2)
        # d03d = llayer_features[0].unsqueeze(2)
        # rtd0 = torch.cat((r03d, t03d, d03d), dim=2)
        # rd1 = self.con30(self.con30(rtd0))
        # rd1_1 = rd1[:, :, 0:1, :, :].squeeze(2)
        # rd1_2 = rd1[:, :, 1:2, :, :].squeeze(2)
        # rd1_3 = rd1[:, :, 2:3, :, :].squeeze(2)
        # # rd1 = rd1_1 + rd1_2 + rd1_3
        # # rd1 = rd1.squeeze(2)
        # rd1 = self.sun4(torch.cat((rd1_1, rd1_2, rd1_3), dim=1))
        rd1 = self.TD1(rlayer_features[0],tlayer_features[0],llayer_features[0])
        rd2 = self.TD2(rlayer_features[1], tlayer_features[1], llayer_features[1])
        rd3 = self.TD3(rlayer_features[2], tlayer_features[2], llayer_features[2])
        rd4 = self.TD4(rlayer_features[3], tlayer_features[3], llayer_features[3])


        # decoder  下行分支

        brd1 = self.TD1_2(self.decbr1(brlayer_features[0]), tlayer_features[0], llayer_features[0])
        brd2 = self.TD2_2(self.decbr2(brlayer_features[1]), tlayer_features[1], llayer_features[1])
        brd3 = self.TD3_2(self.decbr3(brlayer_features[2]), tlayer_features[2], llayer_features[2])
        # print(brlayer_features[3].size())
        brd4 = self.TD4_2(F.interpolate(self.decbr4(brlayer_features[3]), size=12), tlayer_features[3], llayer_features[3])

        gly = tlayer_features[3]
        gll = llayer_features[3]
        glx = rlayer_features[3]
        gly = gly.unsqueeze(2)
        gll = gll.unsqueeze(2)
        glx = glx.unsqueeze(2)
        gl = torch.cat((gly, glx, gll), dim=2)
        gl = self.gl(self.gl(gl))
        gl_1 = gl[:, :, 0:1, :, :].squeeze(2)
        gl_2 = gl[:, :, 1:2, :, :].squeeze(2)
        gl_3 = gl[:, :, 2:3, :, :].squeeze(2)
        gl = torch.cat((gl_1,gl_2,gl_3),dim=1)
        glo = self.glo(gl)

        # glo = F.interpolate(glo, 14, mode='bilinear', align_corners=True)
        # rd44 = F.interpolate(rd4, 14, mode='bilinear', align_corners=True)
        # rd4agg1 = self.xiaoagg1(self.bot1(rd44))
        # res1 = rd4agg1 + glo

        res1 = self.fusion1(rd4, brd4, rd1, glo, glo)
        res2 = self.fusion2(rd3, brd3, res1, glo, rd4)
        res3 = self.fusion3(rd2, brd2, res2, glo, rd3)
        res4 = self.fusion4(rd1, brd1, res3, glo, rd2)

        res = self.b5(res4)
        res = self.end1(res)

        # mul supervision
        out1 = self.sup1(res1)
        out2 = self.sup2(res2)
        out3 = self.sup3(res3)

        # print(res1.size(),out1.size())
        return res, out1, out2, out3


if __name__ == '__main__':
    a = torch.randn(2, 3, 384, 384)
    b = torch.randn(2, 3, 384, 384)
    c = torch.randn(2, 1, 384, 384)
    model = SRAA()
    out = model(a, b, c)
    for i in out:
        print(i.shape)

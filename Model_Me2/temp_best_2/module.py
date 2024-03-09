from math import sqrt

import torch
from mmseg.models.utils.drop import DropPath
from torch import nn

# from backbone.CoTNet.models import trunc_normal_
import torch.nn.functional as F
from 文献代码.FcaNet.model.layer import MultiSpectralAttentionLayer
class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        # print("beforerpooling,",x.size())
        y = self.pool(x) - x
        # print("Afterpool", y.size())
        return y


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FM(nn.Module):
    def __init__(self, in_planes, pool_size=3, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0., drop_path=0.):
        super(FM, self).__init__()
        self.norm1 = norm_layer(in_planes)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(in_planes)
        mlp_hidden_dim = int(in_planes * mlp_ratio)
        self.mlp = Mlp(in_features=in_planes, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()

    def forward(self, x):
        # print("x2",x.size())
        x = x + self.drop_path(self.token_mixer(self.norm1(x)))
        # print("x2",x.size())
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # print("x3",x.size())
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature % 3 == 1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.maxpool = nn.AdaptiveMaxPool2d(1)
        if in_planes != 3:
            hidden_planes = int(in_planes * ratios) + 1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature != 1:
            self.temperature -= 3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x, temperature):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x / temperature, 1)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class Channel_aware_CoordAtt(nn.Module):
    def __init__(self, inp, oup, h, w, reduction=32):
        super(Channel_aware_CoordAtt, self).__init__()
        self.h = h
        self.w = w
        self.pool_h = nn.AdaptiveAvgPool2d((h, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, w))
        self.pool_c = nn.AdaptiveAvgPool2d((w, 1))

        mip = max(8, (inp + self.h) // reduction)

        self.conv1 = nn.Conv2d(inp + self.h, mip, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(inp + self.h, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_y1 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_y2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        channel = x.reshape(n, h, w, c)
        x_c = self.pool_c(channel)

        temp = x_c.permute(0, 2, 1, 3)
        y1 = torch.cat([x_h, temp], dim=1)
        y1 = self.conv1(y1)
        y1 = self.bn1(y1)
        y1 = self.act(y1)

        y2 = torch.cat([x_w, x_c], dim=1)
        y2 = self.conv2(y2)
        y2 = self.bn1(y2)
        y2 = self.act(y2).permute(0, 1, 3, 2)

        y1 = self.conv_y1(y1).sigmoid()

        y2 = self.conv_y2(y2).sigmoid()
        # y2_w = self.conv_y2w(y2_w).sigmoid()

        # 如果下面这个原论文代码用不了的话，可以换成另一个试试
        out = identity * y1 * y2
        # out = a_h.expand_as(x) * a_w.expand_as(x) * identity

        return out


class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        h = self.relu(h)
        h = self.conv2(h)
        return h


class Gru(nn.Module):

    def __init__(self, num_in, num_mid, stride=(1, 1), kernel=1):
        super(Gru, self).__init__()

        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)
        kernel_size = (kernel, kernel)
        padding = (1, 1) if kernel == 3 else (0, 0)
        # reduce dimension
        self.conv_state = BasicConv2d(num_in, self.num_s, kernel_size=kernel_size, padding=padding)
        # generate projection and inverse projection functions
        self.conv_proj = BasicConv2d(num_in, self.num_n, kernel_size=kernel_size, padding=padding)

        self.conv_state2 = BasicConv2d(num_in, self.num_s, kernel_size=kernel_size, padding=padding)
        # generate projection and inverse projection functions
        self.conv_proj2 = BasicConv2d(num_in, self.num_n, kernel_size=kernel_size, padding=padding)

        # reasoning by graph convolution
        self.gcn1 = GCN(num_state=self.num_s, num_node=self.num_n)
        self.gcn2 = GCN(num_state=self.num_s, num_node=self.num_n)
        # fusion
        self.fc_2 = nn.Conv2d(num_in, num_in, kernel_size=kernel_size, padding=padding, stride=(1, 1),
                              groups=1, bias=False)
        self.blocker = nn.BatchNorm2d(num_in)

    def forward(self, x, y):
        batch_size = x.size(0)
        x_state_reshaped = self.conv_state(x).view(batch_size, self.num_s, -1)
        y_proj_reshaped = self.conv_proj(y).view(batch_size, self.num_n, -1)
        x_state_2 = self.conv_state2(x).view(batch_size, self.num_s, -1)
        x_n_state1 = torch.bmm(x_state_reshaped, y_proj_reshaped.permute(0, 2, 1))
        x_n_state2 = x_n_state1 * (1. / x_state_reshaped.size(2))
        x_n_rel1 = self.gcn1(x_n_state2)
        x_n_rel2 = self.gcn2(x_n_rel1)
        x_state_reshaped = torch.bmm(x_n_rel2.permute(0, 2, 1), x_state_2)
        x_state = x_state_reshaped.view(batch_size, 96, 8, 8)
        out = x + self.blocker(self.fc_2(x_state)) + y
        return out


def patch_split(input, bin_size):
    """
    b c (bh rh) (bw rw) -> b (bh bw) rh rw c
    """
    B, C, H, W = input.size()
    bin_num_h = bin_size[0]
    bin_num_w = bin_size[1]
    rH = H // bin_num_h
    rW = W // bin_num_w
    out = input.view(B, C, bin_num_h, rH, bin_num_w, rW)
    out = out.permute(0, 2, 4, 3, 5, 1).contiguous()  # [B, bin_num_h, bin_num_w, rH, rW, C]
    out = out.view(B, -1, rH, rW, C)  # [B, bin_num_h * bin_num_w, rH, rW, C]
    return out


def patch_recover(input, bin_size):
    """
    b (bh bw) rh rw c -> b c (bh rh) (bw rw)
    """
    B, N, rH, rW, C = input.size()
    bin_num_h = bin_size[0]
    bin_num_w = bin_size[1]
    H = rH * bin_num_h
    W = rW * bin_num_w
    out = input.view(B, bin_num_h, bin_num_w, rH, rW, C)
    out = out.permute(0, 5, 1, 3, 2, 4).contiguous()  # [B, C, bin_num_h, rH, bin_num_w, rW]
    out = out.view(B, C, H, W)  # [B, C, H, W]
    return out


class GCN_CAM(nn.Module):
    def __init__(self, num_node, num_channel):
        super(GCN_CAM, self).__init__()
        self.conv1 = nn.Conv2d(num_node, num_node, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Linear(num_channel, num_channel, bias=False)

    def forward(self, x):
        # x: [B, bin_num_h * bin_num_w, K, C]
        out = self.conv1(x)
        out = self.relu(out + x)
        out = self.conv2(out)
        return out


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


class CAAM(nn.Module):
    """
    Class Activation Attention Module
    """

    def __init__(self, feat_in, num_classes, bin_size, norm_layer):
        super(CAAM, self).__init__()
        feat_inner = feat_in // 2
        self.norm_layer = norm_layer
        self.bin_size = bin_size
        self.dropout = nn.Dropout2d(0.1)
        self.conv_cam = nn.Conv2d(feat_in, num_classes, kernel_size=1)
        self.pool_cam = nn.AdaptiveAvgPool2d(bin_size)
        self.sigmoid = nn.Sigmoid()

        bin_num = bin_size[0] * bin_size[1]
        self.gcn = GCN_CAM(bin_num, feat_in)
        self.fuse = nn.Conv2d(bin_num, 1, kernel_size=1)
        self.proj_query = nn.Linear(feat_in, feat_inner)
        self.proj_key = nn.Linear(feat_in, feat_inner)
        self.proj_value = nn.Linear(feat_in, feat_inner)

        self.conv_out = nn.Sequential(
            nn.Conv2d(feat_inner, feat_in, kernel_size=1, bias=False),
            norm_layer(feat_in),
            nn.ReLU(inplace=True)
        )
        self.scale = feat_inner ** -0.5
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        # bin_num_h =S
        cam = self.conv_cam(self.dropout(x))  # [B, K, H, W]
        cls_score = self.sigmoid(self.pool_cam(cam))  # [B, K, bin_num_h, bin_num_w]

        residual = x  # [B, C, H, W]
        cam = patch_split(cam, self.bin_size)  # [B, bin_num_h * bin_num_w, rH, rW, K]
        # print("cam", cam.size())
        x = patch_split(x, self.bin_size)  # [B, bin_num_h * bin_num_w, rH, rW, C]

        B = cam.shape[0]
        rH = cam.shape[2]
        rW = cam.shape[3]
        K = cam.shape[-1]
        C = x.shape[-1]
        cam = cam.view(B, -1, rH * rW, K)  # [B, bin_num_h * bin_num_w, rH * rW, K]
        x = x.view(B, -1, rH * rW, C)  # [B, bin_num_h * bin_num_w, rH * rW, C]

        bin_confidence = cls_score.view(B, K, -1).transpose(1, 2).unsqueeze(3)  # [B, bin_num_h * bin_num_w, K, 1]
        pixel_confidence = F.softmax(cam, dim=2)

        local_feats = torch.matmul(pixel_confidence.transpose(2, 3),
                                   x) * bin_confidence  # [B, bin_num_h * bin_num_w, K, C]
        local_feats = self.gcn(local_feats)  # [B, bin_num_h * bin_num_w, K, C]
        global_feats = self.fuse(local_feats)  # [B, 1, K, C]
        global_feats = self.relu(global_feats).repeat(1, x.shape[1], 1, 1)  # [B, bin_num_h * bin_num_w, K, C]

        query = self.proj_query(x)  # [B, bin_num_h * bin_num_w, rH * rW, C//2]
        key = self.proj_key(local_feats)  # [B, bin_num_h * bin_num_w, K, C//2]
        value = self.proj_value(global_feats)  # [B, bin_num_h * bin_num_w, K, C//2]

        aff_map = torch.matmul(query, key.transpose(2, 3))  # [B, bin_num_h * bin_num_w, rH * rW, K]
        aff_map = F.softmax(aff_map, dim=-1)
        out = torch.matmul(aff_map, value)  # [B, bin_num_h * bin_num_w, rH * rW, C]

        out = out.view(B, -1, rH, rW, value.shape[-1])  # [B, bin_num_h * bin_num_w, rH, rW, C]
        out = patch_recover(out, self.bin_size)  # [B, C, H, W]

        out = residual + self.conv_out(out)
        return out, cls_score


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
        self.gcn = GCN_CAM(bin_num, feat_in)
        self.fuse = nn.Conv2d(bin_num, 1, kernel_size=1)

        # 2
        self.gcn_y = GCN_CAM(bin_num, feat_in)
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


class Bottleneck2D(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck2D, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, inplanes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out





class Bottleneck2DC(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck2DC, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

        if self.downsample is not None:
            residual = self.downsample(x)

        # out += residual

        return out


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
        return self.sigmoid(x) * x


class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


class Dp(nn.Module):
    def __init__(self, inchannel1, inchannel, ourchannel, size):
        super(Dp, self).__init__()
        self.bc = int(inchannel / 2)
        self.pre1 = Bottleneck2D(inchannel, self.bc)
        self.pre2 = Bottleneck2D(inchannel, self.bc)
        self.u1 = nn.Conv2d(inchannel1, inchannel, 3, 1, 1)
        self.trans4 = torch.nn.ConvTranspose2d(inchannel, inchannel, 3, 2, 1, 1, 1, True, 1)
        self.sig = nn.Sigmoid()
        self.trans6 = torch.nn.ConvTranspose2d(inchannel1, inchannel, 3, 2, 1, 1, 1, True, 1)
        self.bc1 = nn.Conv2d(inchannel1, inchannel, 3, 1, 1)
        self.trans5 = torch.nn.ConvTranspose2d(inchannel, inchannel, 3, 2, 1, 1, 1, True, 1)
        self.fm1 = FM(in_planes=inchannel)
        self.bot1 = Bottleneck2D(inchannel, self.bc)
        self.bot1_1 = Bottleneck2D(inchannel, self.bc)
        self.ta3 = Channel_aware_CoordAtt(inchannel, inchannel, size, size)
        self.out = BasicConv2d(inchannel, ourchannel, 3, 1, 1)

    def forward(self, res, bc, r, d):
        x1 = self.u1(bc)
        x1 = self.sig(self.trans4(x1))
        res = self.trans6(res)
        # print(x1.size(), res.size())
        res1 = x1 * res + res
        res1 = res1 + self.ta3(r) * r + self.ta3(d) * d
        bc1 = self.sig(self.trans5(self.bc1(bc)))
        res1 = res1 * bc1 + res1
        res1_1 = self.fm1(self.fm1(res1))
        res1_2 = self.bot1(res1)
        res = self.out(self.bot1_1(res1_1 + res1_2))
        return x1, res

class DD1(nn.Module):
    def __init__(self, channel1, channel2, size, padding, dialation):
        super(DD1, self).__init__()
        self.hide = BasicConv2d(channel1, channel2, 3, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(size[1])
        self.padding = padding
        self.dialation = dialation
        self.aug = AugAttentionModule(channel2)
    def forward(self, hign, low):
        # print("highlow", hign.size(), low.size())
        k1 = self.pool(self.hide(hign))
        B1, C1, H1, W1 = k1.size()

        k2 = self.pool2(self.hide(hign))
        B2, C2, H2, W2 = k2.size()

        x_B3, x_C3, x_H3, x_W3 = low.size()

        new = low.clone()
        # print("x3_new", x3_new.size())
        for i in range(1, B1):
            # print("i", i)
            kernel1 = k1[i, :, :, :]
            kernel1 = kernel1.view(C1, 1, H1, W1)  # [320, 1, 4, 4]
            kernel2 = k2[i, :, :, :]
            kernel2 = kernel2.view(C2, 1, H2, W2)  # [320, 1, 4, 4]
            # DDconv
            x3_r1 = F.conv2d(low[i, :, :, :].view(1, C1, x_H3, x_W3), kernel1, stride=1, padding=self.padding[0],
                             dilation=self.dialation[0],
                             groups=C1)
            x3_r2 = F.conv2d(low[i, :, :, :].view(1, C2, x_H3, x_W3), kernel2, stride=1, padding=self.padding[1],
                             dilation=self.dialation[1],
                             groups=C2)
            new[i, :, :, :] = x3_r1 + x3_r2
        # new = self.aug(new)
        return new
class DD(nn.Module):
    def __init__(self, channel1, channel2, size, size2, padding, dialation):
        super(DD, self).__init__()
        self.hide = BasicConv2d(channel1, channel2, 3, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(size[1])
        self.padding = padding
        self.dialation = dialation
        self.aug = AugAttentionModule(channel2)
        self.dct = MultiSpectralAttentionLayer(channel2, size2[0], size2[1])
    def forward(self, hign, low):
        # print("highlow", hign.size(), low.size())
        k1 = self.pool(self.hide(hign))
        B1, C1, H1, W1 = k1.size()

        k2 = self.pool2(self.hide(hign))
        B2, C2, H2, W2 = k2.size()
        # 0413#

        low = self.dct(low)

        x_B3, x_C3, x_H3, x_W3 = low.size()

        new = low.clone()
        # print("x3_new", x3_new.size())
        for i in range(1, B1):
            # print("i", i)
            kernel1 = k1[i, :, :, :]
            kernel1 = kernel1.view(C1, 1, H1, W1)  # [320, 1, 4, 4]
            kernel2 = k2[i, :, :, :]
            kernel2 = kernel2.view(C2, 1, H2, W2)  # [320, 1, 4, 4]
            # DDconv
            x3_r1 = F.conv2d(low[i, :, :, :].view(1, C1, x_H3, x_W3), kernel1, stride=1, padding=self.padding[0],
                             dilation=self.dialation[0],
                             groups=C1)
            x3_r2 = F.conv2d(low[i, :, :, :].view(1, C2, x_H3, x_W3), kernel2, stride=1, padding=self.padding[1],
                             dilation=self.dialation[1],
                             groups=C2)
            new[i, :, :, :] = x3_r1 + x3_r2
        # new = self.aug(new)
        return new


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class Cross(nn.Module):
    def __init__(self, inchannel1):
        super(Cross, self).__init__()
        self.sa1 = SpatialAttention(7)
        self.sa1_2 = SpatialAttention(3)
        self.r_enhance = BasicConv2d(inchannel1, inchannel1, 1)
        self.weup = BasicConv2d(1, inchannel1, 1)
    def forward(self, r, d):
        N, C, H, W = r.shape
        r_enhance = self.r_enhance(r)
        correlation_maps = F.normalize(r_enhance.reshape(N, C, H*W), dim=2)  # shape=[N, N, HW]
        correlation_matrix = torch.matmul(correlation_maps.permute(0, 2, 1),correlation_maps)  # shape=[N, N, N]
        weight_vectors = correlation_matrix.sum(dim=2).softmax(dim=1).view(N, 1, H, W)  # shape=[N, N]
        weight_vectors = self.weup(weight_vectors)
        d_spatial = self.sa1(d) * d + self.sa1_2(d) * d
        i = r * d_spatial + r
        maps = i * weight_vectors
        return maps

class CM(nn.Module):
    def __init__(self, in_ch):
        super(CM, self).__init__()
        self.avg_weight = nn.AdaptiveAvgPool2d(1)
        self.c_mask = nn.Sigmoid()

    def forward(self, x):
        avg_map_c = self.avg_weight(x)
        c_mask = self.c_mask(avg_map_c)
        return c_mask


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)  # !!!!!!!!!!!!!!!0412 去掉
        return x

class asyConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(asyConv, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size,kernel_size), stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
            self.initialize()
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels)

            center_offset_from_origin_border = padding - kernel_size // 2
            ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
            hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)
            if center_offset_from_origin_border >= 0: # yes
                self.ver_conv_crop_layer = nn.Identity()
                ver_conv_padding = ver_pad_or_crop
                self.hor_conv_crop_layer = nn.Identity()
                hor_conv_padding = hor_pad_or_crop
            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                      stride=stride,
                                      padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                      stride=stride,
                                      padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels)


    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input) # yes
            square_outputs = self.square_bn(square_outputs)
            vertical_outputs = self.ver_conv_crop_layer(input)
            vertical_outputs = self.ver_conv(vertical_outputs)
            vertical_outputs = self.ver_bn(vertical_outputs)
            horizontal_outputs = self.hor_conv_crop_layer(input)
            horizontal_outputs = self.hor_conv(horizontal_outputs)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            return square_outputs + vertical_outputs + horizontal_outputs

class IERD(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(IERD, self).__init__()
        self.asy1 = asyConv(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size[0], stride=stride,
                            padding=padding[0], dilation=dilation[0], groups=groups,
                            padding_mode='zeros', deploy=False)
        self.asy2 = asyConv(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size[1], stride=stride,
                            padding=padding[1], dilation=dilation[1], groups=groups,
                            padding_mode='zeros', deploy=False)
    def forward(self, x):
        x1 = self.asy1(x)
        x2 = self.asy2(x)
        x = x1 + x2
        return x

class CF(nn.Module):
    def __init__(self):
        super(CF, self).__init__()
        self.sa1 = SA(7)
        self.sa2 = SA(7)
    def forward(self, r, d):
        rd = self.sa1(r) * d + self.sa2(r) * d
        return rd

class Attentionshallow(nn.Module):
    def __init__(self,inchannels, outchannels, stride):
        super(Attentionshallow, self).__init__()
        self.depth = nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=stride, dilation=1, padding=1, groups=inchannels)
        self.point = nn.Conv2d(inchannels, inchannels, 1)
    def forward(self, x):
        mask = self.point(self.depth(x))
        # print(x.size(), mask.size())
        x = x * mask
        return x
class predictionshallow(nn.Module):
    def __init__(self,inchannels, outchannels, stride):
        super(predictionshallow, self).__init__()
        self.depth = nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=stride, dilation=1, padding=1)
        self.point = nn.Conv2d(outchannels, outchannels, 1)
    def forward(self, x):
        x = self.point(self.depth(x))
        return x


# 空洞卷积
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


# 池化 -> 1*1 卷积 -> 上采样
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        # 上采样
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    # 整个 ASPP 架构


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        # 1*1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        # 多尺度空洞卷积
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 池化
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # 拼接后的卷积
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class Guide(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Guide, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1, 1, 1, True, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.conv(x)
        return x

    # 整个 ASPP 架构


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) #对应Squeeze操作
        y = self.fc(y).view(b, c, 1, 1) #对应Excitation操作
        return y.expand_as(x)




class NewCF(nn.Module):
    def __init__(self, in_channels, in_channels2):
        super(NewCF, self).__init__()
        self.conv1 = BasicConv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = BasicConv2d(in_channels, in_channels, 3, 1, 1)
        self.conv3 = BasicConv2d(in_channels*2, in_channels, 3, 1, 1)
        self.conv4 = BasicConv2d(in_channels2, in_channels, 3, 1, 1)
        self.sa = SA(7)

    def forward(self, r, d , hr, hd, flag):
        if flag == True:
            h1 = self.conv4(F.interpolate(hr + hd, scale_factor=2))
        else:
            h1 = self.conv4(hr + hd)
        r = self.conv1(h1 * r)
        d = self.conv1(h1 * d)
        h1 = self.sa(h1)
        r = r * h1 + r
        d = d * h1 + d
        r = self.conv3(torch.cat((r, d), dim=1))
        return r

class RA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RA, self).__init__()
        self.convert = nn.Conv2d(in_channels, out_channels, 1)
        self.convs = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channels, 1, 3, padding=1)
        )
        self.channel = out_channels
        # x----low layer feature--->for bei
        # y----high layer feature -->for get attetnion mask
    def forward(self, x, y):
        a = torch.sigmoid(-y)
        x = self.convert(x)
        x = a.expand(-1, self.channel, -1, -1).mul(x)
        y = y + self.convs(x)
        return y

# from DCFM
class AugAttentionModule(nn.Module):
    def __init__(self, input_channels=512):
        super(AugAttentionModule, self).__init__()
        self.query_transform = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
        )
        self.key_transform = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
        )
        self.value_transform = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
        )
        self.scale = 1.0 / (input_channels ** 0.5)
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        B, C, H, W = x.size()
        x = self.conv(x)
        x_query = self.query_transform(x).view(B, C, -1).permute(0, 2, 1)  # B,HW,C
        # x_key: C,BHW
        x_key = self.key_transform(x).view(B, C, -1)  # B, C,HW
        # x_value: BHW, C
        x_value = self.value_transform(x).view(B, C, -1).permute(0, 2, 1)  # B,HW,C
        attention_bmm = torch.bmm(x_query, x_key)*self.scale # B, HW, HW
        attention = F.softmax(attention_bmm, dim=-1)
        attention_sort = torch.sort(attention_bmm, dim=-1, descending=True)[1]
        attention_sort = torch.sort(attention_sort, dim=-1)[1]
        #####
        attention_positive_num = torch.ones_like(attention).cuda()
        attention_positive_num[attention_bmm < 0] = 0
        att_pos_mask = attention_positive_num.clone()
        attention_positive_num = torch.sum(attention_positive_num, dim=-1, keepdim=True).expand_as(attention_sort)
        attention_sort_pos = attention_sort.float().clone()
        apn = attention_positive_num-1
        attention_sort_pos[attention_sort > apn] = 0
        attention_mask = ((attention_sort_pos+1)**3)*att_pos_mask + (1-att_pos_mask)
        out = torch.bmm(attention*attention_mask, x_value)
        out = out.view(B, H, W, C).permute(0, 3, 1, 2)
        return out+x
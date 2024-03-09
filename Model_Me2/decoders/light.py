import torch.nn as nn
import torch.nn.functional as F
import torch


class lightdecoder(nn.Module):
    def __init__(self, inc):
        super(lightdecoder, self).__init__()
        self.conv1 = nn.Conv2d(inc[3], inc[0], 1)
        self.conv2 = nn.Conv2d(inc[2], inc[0], 1)
        self.conv3 = nn.Conv2d(inc[1], inc[0], 1)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, fea_list):
        b, c, h, w = fea_list[0].shape
        f1 = F.interpolate(self.conv1(fea_list[3]), size=(h, w), mode='nearest')
        f2 = F.interpolate(self.conv2(fea_list[2]), size=(h, w), mode='nearest')
        f3 = F.interpolate(self.conv3(fea_list[1]), size=(h, w), mode='nearest')
        # fea_list = [F.interpolate(fea, size=(h, w), mode='nearest') for fea in fea_list[1:]]
        p4 = f1 + f2 + f3 + fea_list[0]
        # p4 = fea_list[0] + self.up(fea_list[1] + self.up(fea_list[2] + self.up(fea_list[3])))
        return p4

import torch.nn as nn


class unet(nn.Module):
    def __init__(self, inc):
        super(unet, self).__init__()
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv1 = nn.Conv2d(inc[3], inc[2], 3, 1, 1)
        self.conv2 = nn.Conv2d(inc[2], inc[1], 3, 1, 1)
        self.conv3 = nn.Conv2d(inc[1], inc[0], 3, 1, 1)
        # self.conv4 = nn.Conv2d(inc[0], inc[0], 3, 1, 1)

    def forward(self, fea_list):
        p1 = self.conv1(fea_list[3])
        p2 = self.conv2(fea_list[2] + self.up2(p1))
        p3 = self.conv3(fea_list[1] + self.up2(p2))
        p4 = fea_list[0] + self.up2(p3)

        return p4

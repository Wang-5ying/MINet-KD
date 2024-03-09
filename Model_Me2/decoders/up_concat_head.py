import torch.nn as nn
import torch
import torch.nn.functional as F


class UpConcatHead(nn.Module):
    def __init__(self, inc, embedding_dim=512, num_classes=41, **kwargs):
        super(UpConcatHead, self).__init__()

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=sum(inc), out_channels=embedding_dim, kernel_size=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU())

        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, inputs):
        # inputs = self._transform_inputs(inputs)
        inputs = [F.interpolate(
            level,
            size=inputs[0].shape[2:],
            mode='bilinear',
        ) for level in inputs]

        inputs = torch.cat(inputs, dim=1)
        x = self.linear_fuse(inputs)
        # x = self.dropout(x)
        x = self.linear_pred(x)
        return x

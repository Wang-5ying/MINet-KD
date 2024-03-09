'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from decode_head import BaseDecodeHead


class CLS(BaseDecodeHead):
    def __init__(self,
                 in_channels,
                 aff_channels=512,
                 aff_kwargs=dict(),
                 **kwargs):
        super(CLS, self).__init__(
            in_channels=in_channels,input_transform='multiple_select', **kwargs)
        self.aff_channels = aff_channels
        self.in_channels = in_channels
        self.input_transform = 'resize_concat'
        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.align = ConvModule(
            self.aff_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)[0]

        x = self.squeeze(inputs)
        print(x.shape)
        output = self.cls_seg(x)
        return output

if __name__ == '__main__':
    inputs = [torch.randn(2, 92, 58, 58), torch.randn(2, 192, 158, 158), torch.randn(2, 214, 258, 258), torch.randn(2, 214, 258, 258)]
    moeld = CLS(in_channels=[92, 192, 214, 214], channels=92, num_classes=41)
    out = moeld(inputs)
    print(out.shape)

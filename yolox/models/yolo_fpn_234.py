#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv


class YOLOXFPN_234(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark2", "dark3", "dark4", "dark5"),
        # in_features=("dark3", "dark4", "dark5"),
        in_channels=[128, 256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, out_features=in_features, depthwise=depthwise, act=act)
        # self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        
        self.lateral_conv0 = BaseConv(
            int(in_channels[3] * width), int(in_channels[2] * width), 1, 1, act=act
        )
        self.out4 = CSPLayer(
            int(2 * in_channels[2] * width),    # 1024
            int(in_channels[2] * width),        #  512
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.out3 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.xlarge_conv2 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.out2 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )


    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x3, x2, x1, x0] = features    # C2,C3,C4,C5

        x1_in = self.lateral_conv0(x0)          # 1024->512/32  C5
        x1_in = self.upsample(x1_in)            # 512/16        C4
        x1_in = torch.cat([x1_in, x1], 1)       # 1024/16       C4
        out_dark4 = self.out4(x1_in)            # 512/16        F4

        x2_in = self.reduce_conv1(out_dark4)    # 512->256/16   C4
        x2_in = self.upsample(x2_in)            # 256/8         C3
        x2_in = torch.cat([x2_in, x2], 1)       # 512/8         C3
        out_dark3 = self.out3(x2_in)            # 256/8         F3

        x3_in = self.xlarge_conv2(out_dark3)    # 256/8->128/8  C3
        x3_in = self.upsample(x3_in)            # 128/4         C2
        x3_in = torch.cat([x3_in, x3], 1)       # 256/4         C2
        out_dark2 = self.out2(x3_in)            # 128/4         F2      

        outputs = (out_dark2, out_dark3, out_dark4)    # F2,F3,F4
        return outputs

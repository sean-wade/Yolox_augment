#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN

from .efficient_rep.rep_block import RepVGGBlock
from .efficient_rep import RepPANNeck


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            losses = self.head(
                fpn_outs, targets, x
            )
            if len(losses) == 6:
                loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = losses
                outputs = {
                    "total_loss": loss,
                    "iou_loss": iou_loss,
                    "l1_loss": l1_loss,
                    "conf_loss": conf_loss,
                    "cls_loss": cls_loss,
                    "num_fg": num_fg,
                }
            else:
                loss, iou_loss, conf_loss, cls_loss, l1_loss, attr_loss, num_fg = losses
                outputs = {
                    "total_loss": loss,
                    "iou_loss": iou_loss,
                    "l1_loss": l1_loss,
                    "conf_loss": conf_loss,
                    "cls_loss": cls_loss,
                    "attr_loss":attr_loss,
                    "num_fg": num_fg,
                }

        else:
            outputs = self.head(fpn_outs)

        return outputs
    

    def switch_to_deploy(self):
        if isinstance(self.backbone, RepPANNeck):
            # reparemeter backbone to vgg-like
            for layer in self.backbone.modules():
                if isinstance(layer, RepVGGBlock):
                    layer.switch_to_deploy()
            print("Switch model to deploy modality.")

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from .coco import COCODataset
from .coco_classes import COCO_CLASSES
from .datasets_wrapper import ConcatDataset, Dataset, MixConcatDataset
from .mosaicdetection import MosaicDetection
from .voc import VOCDetection

from .dtld import DTLDDetection, DTLD_CLASSES, DTLD_ATTRIBUTES
from .sgtls import SGTLS_Detection, SGTLS_CLASSES, SGTLS_ATTRIBUTES 

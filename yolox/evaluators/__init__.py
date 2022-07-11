#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from .coco_evaluator import COCOEvaluator
from .voc_evaluator import VOCEvaluator
from .dtld_evaluator import DTLDEvaluator

from .metrics import *
from .dtld_eval import get_metrics

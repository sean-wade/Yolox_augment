import cv2
import torch
import numpy as np

from yolox.exp import Exp, get_exp

exp = get_exp("/home/zhanghao/code/master/2_DET2D/YOLOX/exps/dtld/dtld_s.py", "test_dtld")
evaluator = exp.get_evaluator(batch_size=1, is_distributed=False)

model = exp.get_model()
model.to("cuda")
ckpt = torch.load("YOLOX_outputs/dtld_s2/epoch_1_ckpt.pth", map_location="cpu")
model.load_state_dict(ckpt["model"])

exp.eval(model, evaluator, False)
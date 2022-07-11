import cv2
import numpy as np

from yolox.data.datasets import DTLDDetection
from yolox.exp import Exp, get_exp


dataset = DTLDDetection()



exp = get_exp("/home/zhanghao/code/master/2_DET2D/YOLOX/exps/dtld/dtld_s.py", "test_dtld")
# exp = get_exp("/home/zhanghao/code/master/2_DET2D/YOLOX/exps/yolox_x_dtld.py", "test_voc")

data_loader = exp.get_data_loader(1, False, no_aug=True, cache_img=False)


for img, target, shape, idx in data_loader:
    print(idx, target[0].shape)
    print("---")

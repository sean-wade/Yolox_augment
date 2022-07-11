import os
import shutil
from tqdm import tqdm

src = "/mnt/data/TLS/DTLD_Yolo/images/train/"
des = "/mnt/data/TLS/DTLD_VOC/VOCdevkit/VOC2007/JPEGImages/"


fs = os.listdir(src)

for ff in tqdm(fs):
    src_f = src + ff
    dst_f = des + ff
    if not os.path.exists(dst_f):
        shutil.copy(src_f, dst_f)


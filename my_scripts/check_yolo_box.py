import os
from tqdm import tqdm


label_path = "/mnt/data/TLS/DTLD_VOC/VOCdevkit/VOC2007/labels/"

labels = os.listdir(label_path)

for ll in tqdm(labels):
    lines = open(label_path + ll, "r").readlines()
    for line in lines:
        l,x,y,w,h = [float(m) for m in line.strip().split()]

        if x<0 or y<0:
            print(ll, x, y, "xy<0")

        if w<0 or h<0:
            print(ll, w, h, "wh<0")

        if (x+w)>1 or (y+h)>1:
            print(ll, x+w, y+h, "xy2>1")




import os

files = os.listdir("/mnt/data/TLS/DTLD_Yolo/images/val/")


with open("/mnt/data/TLS/DTLD_VOC/VOCdevkit/VOC2007/ImageSets/Main/val.txt", "w") as ff:
    
    for file_name in files:
        ff.write(file_name[:-4] + "\n")


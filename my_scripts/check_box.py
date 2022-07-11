import xml.etree.ElementTree as ET
import os

import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm

xml_root = "/mnt/data/TLS/DTLD_VOC/VOCdevkit/VOC2007/Annotations/"
new_xml_root = "/mnt/data/TLS/DTLD_VOC/VOCdevkit/VOC2007/new_Annotations"
image_root = "/mnt/data/TLS/DTLD_VOC/VOCdevkit/VOC2007/JPEGImages"

jpg_name_list = sorted(os.listdir(image_root))


def print_all_classes():
    all_name_list = []
    for jpg_name in jpg_name_list:
        xml_name = jpg_name[:-4] + ".xml"
        print(f"{xml_name}")
        xml_path = os.path.join(xml_root, xml_name)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name = obj.find("name").text
            all_name_list.append(name)
        print(all_name_list)


def check_hw():
    tranposed_name_lists = []
    for jpg_name in jpg_name_list:
        xml_name = jpg_name[:-4] + ".xml"
        xml_path = os.path.join(xml_root, xml_name)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        image_path = os.path.join(image_root, jpg_name)
        img = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
        h, w, _ = img.shape
        if height != h or width != w:
            print(width, w, height, h)
            print(f"{xml_name}'s h, w is tranposed.")
            tranposed_name_lists.append(xml_name)
    print(tranposed_name_lists)


def check_bbox():
    if not os.path.exists(new_xml_root):
        os.makedirs(new_xml_root)

    for jpg_name in tqdm(jpg_name_list):
        xml_path = os.path.join(xml_root, jpg_name[:-4] + ".xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        
        image_path = os.path.join(image_root, jpg_name)
        img = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
        h, w, _ = img.shape
        
        size.find("width").text = str(w)
        size.find("height").text = str(h)
        
        folder = root.find('folder')
        filename = root.find('filename')
        path = root.find('path')
        
        if path == None or folder == None or filename == None:
            #print("None path : ", image_path)
            pass
        else:
            folder.text = 'VOCdevkit'
            filename.text = image_path[:-4]
            path.text = image_path
        
        for obj in root.findall("object"):
            bnd_box = obj.find("bndbox")
            bbox = [
                int(float(bnd_box.find("xmin").text)),
                int(float(bnd_box.find("ymin").text)),
                int(float(bnd_box.find("xmax").text)),
                int(float(bnd_box.find("ymax").text)),
            ]
            
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                print("bbox[0] >= bbox[2] or bbox[1] >= bbox[3]", bbox, jpg_name)
                root.remove(obj)
            
            if bbox[3] > h:
                print(jpg_name, "bbox[3] >= h", bbox, h)
                bnd_box.find("ymax").text = str(h)
            
            if bbox[2] > w:
                bnd_box.find("xmax").text = str(w)
                print(jpg_name, "bbox[2] >= w", bbox, w)
                                 
            '''
            elif bbox[3] > h or bbox[2] > w:
            
                #bnd_box.find("xmax").text = str(min(w, bbox[2]))
                #bnd_box.find("ymax").text = str(min(h, bbox[3]))
                print(jpg_name, "bbox[3] > h or bbox[2] > w", bbox, h, w, width, height)
                w_scale = w / width  #normaly, width is bigger
                h_scale = h / height 
                x1 = int(bbox[0] * w_scale)
                x2 = int(bbox[2] * w_scale)
                y1 = int(bbox[1] * h_scale)
                y2 = int(bbox[3] * h_scale)
                bnd_box.find("xmin").text = str(x1)
                bnd_box.find("xmax").text = str(x2)
                bnd_box.find("ymin").text = str(y1)
                bnd_box.find("ymax").text = str(y2)
                print("    new x1, y1, x2, y2: ", x1, y1, x2, y2)
                
                #size.find("width").text = str(w)
                #size.find("height").text = str(h)
                
                if x2 > w or y2 > h:
                    print("    still wrong ... remove it...")
                    root.remove(obj)    
            '''    
        tree.write(os.path.join(new_xml_root, jpg_name[:-4] + ".xml"))

check_bbox()
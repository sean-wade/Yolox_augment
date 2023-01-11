import os
import json
import sys

import cv2
import torch
import numpy as np
from loguru import logger

from .datasets_wrapper import Dataset


__author__ = "Sean Wade"
__maintainer__ = "Sean Wade"


SGTLS_CLASSES = ("tl",)    # traffic_light
SGTLS_ATTRIBUTES = {
        "direction"     : ["front", "left", "right", "back"], 
        "orientation"   : ["horizontal", "vertical", "square"], 
        "state"         : ["on", "off", "unknown"], 
        "indication"    : ["motor", "non_motor", "pedestrian", "digit", "other", "unknown"], 
        "occlusion"     : ["not_occluded", "slight_occluded", "half_occluded", "heavily_occluded"], 
        "truncation"    : ["not_truncated", "truncated"],
        "blur"          : ["not_blur", "blur"],
        "child_num"     : ["one", "two", "three", "four", "unknown"],
        "relevance"     : ["not_relevant", "relevant"], 

        "color"         : ["red", "green", "yellow", "dark", "unknown"],
        "pict"          : ["circle", "arrow_straight", "arrow_left", "arrow_right", "arrow_uturn", 
                           "arrow_straight_left", "arrow_straight_right", "arrow_uturn_left", 
                           "bicycle", "pedestrian", 
                           "lane_stop", "lane_straight", 
                           "digit", 
                           "unknown"
                          ]
    }

# ['direction', 'orientation', 'state', 'indication', 'occlusion', 'truncation', 'blur', 'child_num', 'relevance', 'color', 'pict']

num_int = {
    "one" : 1, 
    "two" : 2,
    "three" : 3,
    "four" : 4,
    "unknown" : -1,
}


def get_obj_color(obj):
    if obj["state"] == "unknown":
        return "unknown"

    elif obj["state"] == "off":
        return "dark"
    
    else:
        child_num = num_int.get(obj["child_num"], obj["child_num"])
        if child_num < 0:
            # print("    **** Error, child_num unknown !")
            return "unknown"
        else:
            color_nums = np.array([obj["child_color"].count(cc) for cc in SGTLS_ATTRIBUTES["color"]])
            if color_nums[:3].sum() > 0:
                # r/g/y
                return SGTLS_ATTRIBUTES["color"][np.argmax(color_nums[:3])]
            elif color_nums[3] > 0:
                return "dark"
            else:
                return "unknown"


def get_obj_pict(obj):
    if obj["state"] in ["unknown", "off"]:
        return "unknown"

    child_num = num_int.get(obj["child_num"], obj["child_num"])
    if child_num < 0:
        # print("    **** Error, child_num unknown !")
        return "unknown"
    
    new_child_shapes = obj["child_shape"].copy()
    for i,new_child_shape in enumerate(new_child_shapes):
        if new_child_shape.startswith("digit"):
            new_child_shapes[i] = "digit"

    pict_counts = np.array([new_child_shapes.count(cc) for cc in SGTLS_ATTRIBUTES["pict"]])
    if pict_counts[-1] == child_num:
        # all unknown
        return "unknown"

    pict_counts[-1] = -1
    max_idx = np.argmax(pict_counts)
    max_num = pict_counts[max_idx]
    if max_num == 0:
        return "unknown"

    # 可能出现: 如4个子灯中，第一个是左转，第四个是倒计时
    # 但是目前标注公司导出的文件有问题，第四个灯的信息没有导出
    pict_counts[max_idx] = -1
    second_maxidx = np.argmax(pict_counts)
    second_max_num = pict_counts[second_maxidx]
    if second_max_num > 0:
        print("    **** Warning, at least two pict !!!")

    return SGTLS_ATTRIBUTES["pict"][max_idx]


class SGTLS_Transform(object):

    """Transforms a SG-TLS annotation into a Tensor of bbox coords and label index and attributes.
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        height (int): height
        width (int): width
    """

    def __init__(self):
        self.attr_dict = SGTLS_ATTRIBUTES


    def __call__(self, annos):
        """
        Arguments:
            annos (annotation) : the annotation of SG-TLS.
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name, attributes]
        """
        res = np.empty((0, 16))

        for obj in annos["objects"]:
            # if obj["class"] != SGTLS_CLASSES[0] or obj["indication"] == "digit":
            #     continue

            bndbox = [obj["bbox"][0], obj["bbox"][1], obj["bbox"][2], obj["bbox"][3]]
            bndbox.append(0)    # class_id (0)
            
            # 通过联合判断子灯状态, 增加 color 和 pict 属性
            if "color" not in obj:
                color = get_obj_color(obj)
                obj.update({"color" : color})

            if "pict" not in obj:
                pict = get_obj_pict(obj)
                obj.update({"pict" : pict})

            for attr_name, attr_list in self.attr_dict.items():
                if "child_num" == attr_name and obj[attr_name] not in num_int:
                    bndbox.append(obj[attr_name] - 1)
                else:
                    bndbox.append(attr_list.index(obj[attr_name]))

            res = np.vstack((res, bndbox))  # [xmin, ymin, xmax, ymax, label_ind, ori, occ, rel, dir, sta, asp, pic]

        return res


class SGTLS_Detection(Dataset):
    def __init__(
        self, 
        json_paths,
        img_size=(576, 1024),
        preproc=None,
        target_transform=SGTLS_Transform(),
        dataset_name="SGTLS_train",
        cache=False,
    ):
        super().__init__(img_size)
        self.json_paths = json_paths if isinstance(json_paths, list) else [json_paths]
        

        self.load_json_data()
        assert self.check_paths_time_consistency(), "Error, dataset cannot pass check_paths_time_consistency !"

        self.img_size = img_size
        self.preproc = preproc
        self.target_transform = target_transform
        self.name = dataset_name

        self.annotations = self._load_all_annotations()
        self.imgs = None

        self.attr_nums = len(SGTLS_ATTRIBUTES.keys())

        if cache:
            self._cache_images()
           
        logger.info("[%s]-dataset init done, total [%d] images and [%d] labels."%(self.name, self.img_nums, len(self.annotations)))


    def load_json_data(self):
        self.image_paths = []
        self.anno_paths = []
        self.img_nums = 0
        for json_pp in self.json_paths:
            assert os.path.exists(json_pp), "%s does not exist!"%json_pp
            data = json.load(open(json_pp))
            self.image_paths.extend(data["images"])
            self.anno_paths.extend(data["annos"])
            self.img_nums += len(data["annos"])
            self.data_path = os.path.dirname(json_pp)  # os.path.abspath(os.path.join(json_pp, ".."))
        print("cache path = %s"%self.data_path)


    def check_paths_time_consistency(self):
        if len(self.image_paths) != len(self.anno_paths):
            print("Error, paths length aren't the same %d, %d ! " % (len(self.image_paths), len(self.anno_paths)))
            return False
        for dp,ap in zip(self.image_paths, self.anno_paths):
            name1 =  os.path.basename(dp)
            name2 = os.path.basename(ap)
            if ".".join(name1.split('.')[:2]) != ".".join(name2.split('.')[:2]):
                print("Error, different timestamp between %s and %s" %(name1, name2))
                return False
        return True


    def __len__(self):
        return self.img_nums


    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have 40G+ RAM and 40G+ available disk space for training SGTLS.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = os.path.join(self.data_path, f"img_resized_cache_{self.name}_{max_h}_{max_w}.array")
        if not os.path.exists(cache_file):
            logger.info(
                "Caching images for the first time. This might take about x minutes for SGTLS"
            )
            self.imgs = np.memmap(
                cache_file,
                shape=(self.img_nums, max_h, max_w, 3),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self.load_resized_img(x),
                range(len(self.annotations)),
            )
            pbar = tqdm(enumerate(loaded_images), total=self.img_nums)
            for k, out in pbar:
                self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!\n"
                "Everytime the self.input_size is changed in your exp file, you need to delete\n"
                "the cached data and re-generate them.\n"
            )

        logger.info("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(self.img_nums, max_h, max_w, 3),
            dtype=np.uint8,
            mode="r+",
        )


    def _load_all_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in range(self.img_nums)]


    def load_anno_from_ids(self, index):
        anno_file = self.anno_paths[index]
        # print(anno_file)
        cur_annos = json.load(open(anno_file))
        assert self.target_transform is not None
        res = self.target_transform(cur_annos)

        img_info = (cur_annos["infos"]["image_height"], cur_annos["infos"]["image_width"])
        height, width = img_info

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r
        resized_info = (int(height * r), int(width * r))

        return (res, img_info, resized_info)


    def load_anno(self, index):
        return self.annotations[index][0]


    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        return resized_img


    def load_image(self, index):
        img_file = self.image_paths[index]
        img = cv2.imread(img_file)
        assert img is not None, f"file named {img_file} not found"
        return img


    def pull_item(self, index):
        """Returns the original image and target at an index for mixup

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            img, target
        """
        if self.imgs is not None:
            target, img_info, resized_info = self.annotations[index]
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)
            target, img_info, _ = self.annotations[index]

        return img, target, img_info, index


    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        return img, target, img_info, img_id


    def evaluate_detections(self, pred_boxes, save_path, with_attr=False):
        from yolox.evaluators import get_metrics

        gt_boxes = {}
        for idx, anno_info in enumerate(self.annotations):
            target, img_info, resized_info = anno_info
            r = min(self.img_size[0] / img_info[0], self.img_size[1] / img_info[1])
            gt_boxes_idx = target[:,:4] / r
            labels_idx = target[:,4].reshape(-1,1)
            if with_attr:
                gt_attr = target[:,-len(SGTLS_ATTRIBUTES):]
                gt_boxes[idx] = np.hstack((labels_idx, gt_boxes_idx, gt_attr))
            else:
                gt_boxes[idx] = np.hstack((labels_idx, gt_boxes_idx))
        
        return get_metrics(pred_boxes, gt_boxes, save_path, with_attr=with_attr, attr_num=self.attr_nums)


if __name__ == "__main__":
    sgtls = SGTLS_Detection(json_paths = ["/home/jovyan/workspace/tl_infos/000_010/total_train_infos.json",
                                          "/home/jovyan/workspace/tl_infos/011/011_20220919_train_infos.json",
                                          "/home/jovyan/workspace/tl_infos/013_014/total_013_014_train_infos.json"]
                           )
    # for i in range(len(sgtls)):
    #     img, target, img_info, idx = sgtls.pull_item(i)
    #     print(target)



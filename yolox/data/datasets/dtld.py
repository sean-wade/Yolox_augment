import os
import os.path

import cv2
import numpy as np
from loguru import logger

from .driveu_dataset import DriveuDatabase
from .datasets_wrapper import Dataset


DTLD_CLASSES = ("traffic_light",)
DTLD_ATTRIBUTES = {
        "orientation"   : ["vertical", "horizontal"], 
        "occlusion"     : ["occluded", "not_occluded"], 
        "relevance"     : ["relevant", "not_relevant"], 
        "reflection"    : ["reflected", "not_reflected"], 
        "direction"     : ["front", "back", "left", "right"], 
        "state"         : ["red", "green", "yellow", "red_yellow", "off", "unknown"], 
        "aspects"       : ["one_aspect", "two_aspects", "three_aspects", "four_aspects", "unknown"], 
        "pictogram"     : ["circle", "arrow_left", "arrow_right", "arrow_straight", 
                            "tram", "pedestrian", "bicycle", "unknown",
                            "pedestrian_bicycle", "arrow_straight_left"]
    }


class DTLDTransform(object):

    """Transforms a DTLD annotation into a Tensor of bbox coords and label index and attributes.
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        height (int): height
        width (int): width
    """

    def __init__(self):
        self.attr_dtct = DTLD_ATTRIBUTES


    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the objects of DriveuObject.
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name, attributes]
        """
        res = np.empty((0, 13))
        for obj in target:
            bndbox = [obj.x, obj.y, obj.x+obj.width, obj.y+obj.height, 0]
            
            for attr_name, attr_list in self.attr_dtct.items():
                bndbox.append(attr_list.index(obj.attributes[attr_name]))

            res = np.vstack((res, bndbox))  # [xmin, ymin, xmax, ymax, label_ind, ori, occ, rel, dir, sta, asp, pic]

        return res


class DTLDDetection(Dataset):

    """
    DTLD Detection Dataset Object

    input is image, target is annotation

    Args:
        data_dir (string): filepath to DTLD folder.
        json_path (string): json file to use (from DTLD dataset)
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
    """

    def __init__(
        self,
        data_dir="/mnt/data/TLS/DTLD",
        json_path="/mnt/data/TLS/DTLD/DTLD_Labels_v2.0/v2.0/Fulda.json",
        img_size=(416, 416),
        preproc=None,
        target_transform=DTLDTransform(),
        dataset_name="DTLD_train",
        cache=False,
    ):
        super().__init__(img_size)
        self.root = data_dir
        self.json_path = json_path
        self.img_size = img_size
        self.preproc = preproc
        self.target_transform = target_transform

        self._classes = ("traffic_light",)

        self.root = data_dir
        self.database = DriveuDatabase(json_path)
        if not self.database.open(data_dir):
            raise "Error opening database path : [%s] !!!"%self.root

        self.annotations = self._load_coco_annotations()
        self.imgs = None
        self.name = dataset_name

        if cache:
            self._cache_images()


    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have 60G+ RAM and 30G available disk space for training DTLD.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = os.path.join(self.root, f"img_resized_cache_{self.name}_{max_h}_{max_w}.array")
        if not os.path.exists(cache_file):
            logger.info(
                "Caching images for the first time. This might take about 3 minutes for DTLD"
            )
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.database.images), max_h, max_w, 3),
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
            pbar = tqdm(enumerate(loaded_images), total=len(self.annotations))
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
            shape=(len(self.database.images), max_h, max_w, 3),
            dtype=np.uint8,
            mode="r+",
        )


    def __len__(self):
        return len(self.database.images)


    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in range(len(self.database.images))]


    def load_anno_from_ids(self, index):
        
        target = self.database.images[index].objects
        assert self.target_transform is not None
        res = self.target_transform(target)

        img_info = (1024, 2048)
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
        _, img = self.database.images[index].get_image()
        assert img is not None, f"file named {self._imgpath % self.database.images[index].file_path} not found"
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
                gt_attr = target[:,-8:]
                gt_boxes[idx] = np.hstack((labels_idx, gt_boxes_idx, gt_attr))
            else:
                gt_boxes[idx] = np.hstack((labels_idx, gt_boxes_idx))
            
        return get_metrics(pred_boxes, gt_boxes, save_path, with_attr=with_attr)

"""
The scripts is from DTLD-dataset.
For details, please visit github.
"""
from __future__ import print_function

import copy
import json
import logging
import os
import sys

import cv2
import numpy as np



__author__ = "Andreas Fregin, Julian Mueller and Klaus Dietmayer"
__maintainer__ = "Julian Mueller"
__email__ = "julian.mu.mueller@daimler.com"

"""
DTLD_v1 suppports only .yml files
DTLD_v2(04/2020) only supports .json files
"""


# attributes string to plot on image (original attributes is too long)
attr_plot_str = {
    "one_aspect" : "1",
    "two_aspects" : "2",
    "three_aspects" : "3",
    "four_aspects" : "4",
    
    "front" : "FR",
    "back" : "BK",
    "left" : "LF",
    "right" : "RT",

    "horizontal" : "HOR",
    "vertical" : "VER",

    "not_occluded" : "N_OC",
    "occluded" : "OC",

    "circle" : "O",
    "arrow_left" : "<-",
    "arrow_right" : "->",
    "arrow_straight" : "^",
    "tram" : "TRAM",
    "bicycle" : "BICY",
    "pedestrian" : "PED",

    "not_reflected" : "N_REF",
    "reflected" : "REF",


    "off" : "OFF",
    "red" : "RED",
    "yellow" : "YEL",
    "red_yellow" : "RED_Y",
    "green" : "GRE",
    "unknown" : "UNK",

    "not_relevant" : "N_REL",
    "relevant" : "REL"

}

colors = [tuple(map(int, np.random.randint(150, high=255, size=(3,)))) for _ in range(100)]

def plot_attributes(img, obj, color=(0,255,255)):
    attr = obj.attributes
    
    cv2.rectangle(img, (obj.x, obj.y+obj.height+1), ((obj.x+35, obj.y+obj.height+110)), color)

    for idx, value in enumerate(attr.values()):
        cv2.putText(img, attr_plot_str[value], (obj.x, obj.y+obj.height+13*idx+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return img


class DriveuObject:
    """
    Class holding properties of a label object in the dataset

    Attributes:
        x(int):             X coordinate of upper left corner of bbbox label
        y(int):             Y coordinate of upper left corner of bbox label
        width(int):         Width of bounding box label
        height(int):        Height of bounding box label
        attributes(dict):   Attributes dictionary with label attributes
                            "direction" =   ["front" || "back" || "left"
                            || "right"],
                            "occlusion" = ["occluded" || "not_occluded"],
                            "relevance" = ["relevant" || "not_relevant"],
                            "orientation" = ["vertical" || "horizontal"],
                            "aspects" = ["one_aspect" || "two_aspects",
                            || "three_aspects" || "four_aspects" || "unknown"],
                            "state" = ["red" || "green" || "yellow"
                            || "red_yellow" || "off" || "unknown"]
                            "pictogram" = ["circle" || "arrow_left"
                            || "arrow_right" || "arrow_straight" || "tram"
                            || "pedestrian" || "bicycle" || unknown]
        unique_id(int):  Unique ID of the object
        track_id(string):Track ID of the objec (representing one real-world
        TL instance)
    """

    def __init__(self):
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0
        self.attributes = {}
        self.unique_id = 0
        self.track_id = 0

    def parse_object_dict(self, object_dict: dict):
        """
        Method loading label data from json file dict

        Args:
            object_dict(dict): label dictionary read from json file
        """
        self.x = object_dict["x"]
        self.y = object_dict["y"]
        self.width = object_dict["w"]
        self.height = object_dict["h"]
        self.attributes = object_dict["attributes"]
        self.unique_id = object_dict["unique_id"]
        self.track_id = object_dict["track_id"]

    def color_from_attributes(self):
        """
        Return color for state of class identity

        Returns:
            Color-vector (BGR) for traffic light visualization
        """
        # Second last digit indicates state/color
        if self.attributes["state"] == "red":
            return (0, 0, 255)
        elif self.attributes["state"] == "yellow":
            return (0, 255, 255)
        elif self.attributes["state"] == "red_yellow":
            return (0, 165, 255)
        elif self.attributes["state"] == "green":
            return (0, 255, 0)
        else:
            return (255, 255, 255)


class DriveuImage:
    """
    Class holding properties of one image in the DriveU Database

    Attributes:
        file_path (string):         Path of the left camera image
        disp_file_path (string):    Path of the corresponding disparity image
        timestamp (float):          Timestamp of the image

        objects (DriveuObject)     : Labels in that image
    """

    def __init__(self):
        self.file_path = ""
        self.disp_file_path = ""
        self.timestamp = 0
        self.objects = []

    def parse_image_dict(self, image_dict: dict, data_base_dir: str = ""):
        """
        Method loading image data from json file dict

        Args:
            image_dict(dict): image dictionary read from json label file
            data_base_dir(str): optional, if file paths are oudated
            (DTLD was moved from directory). Note that the internal
            DTLD should not be changed!
        """
        # Parse images
        if data_base_dir != "":
            inds = [i for i, c in enumerate(image_dict["image_path"]) if c == "/"]
            self.file_path = os.path.join(data_base_dir,
                                          image_dict["image_path"][inds[-4]:].strip("/"))
            inds = [
                i for i, c in enumerate(image_dict["disparity_image_path"]) if c == "/"
            ]
            self.disp_file_path = os.path.join(data_base_dir,
                                        image_dict["disparity_image_path"][inds[-4]:].strip("/"))

        else:
            self.file_path = image_dict["image_path"]
            self.disp_file_path = image_dict["disparity_image_path"]
            self.timestamp = image_dict["time_stamp"]

        # Parse labels
        for o in image_dict["labels"]:
            label = DriveuObject()
            label.parse_object_dict(o)
            self.objects.append(label)


    def get_image(self):
        """
        Method loading the left unrectified color image in 8 bit

        Returns:
            (bool np.array): (status, 8 Bit BGR color image)
        """
        if os.path.isfile(self.file_path):
            # Load image from file path, do debayering and shift
            img = cv2.imread(self.file_path, cv2.IMREAD_UNCHANGED)
            if os.path.splitext(self.file_path)[1] == ".tiff":
                img = cv2.cvtColor(img, cv2.COLOR_BAYER_GB2BGR)
                # Images are saved in 12 bit raw -> shift 4 bits
                img = np.right_shift(img, 4)
                img = img.astype(np.uint8)

            return True, img

        else:
            logging.error("Image {} not found. Please check image file paths!".format(self.file_path))
            sys.exit(1)
            return False, np.array()


    def get_labeled_image(self, with_attr=False):
        """
        Method loading the left unrectified color image with visualized labels

        Returns:
            Labeled 8 Bit BGR color image
        """

        status, img = self.get_image()

        if status:
            for i,o in enumerate(self.objects):
                cv2.rectangle(
                    img,
                    (o.x, o.y),
                    (o.x + o.width, o.y + o.height),
                    o.color_from_attributes(),
                    2,
                )
        
                if with_attr:
                    plot_attributes(img, o, colors[i])

        return img



class DriveuDatabase:
    """
    Class describing the DriveU Dataset containing a list of images

    Attributes:
        images (List of DriveuImage)  All images of the dataset
        file_path (string):           Path of the dataset (.json)
    """

    def __init__(self, file_path):
        self.images = []
        self.file_path = file_path

    def open(self, data_base_dir: str = ""):
        """
        Method loading the dataset

        Args:
            data_base_dir(str): Base path where images are stored, optional
            if image paths in json are outdated
        """

        if os.path.exists(self.file_path) is not None:
            label_file_extension = os.path.splitext(self.file_path)[1]
            if label_file_extension == ".json":
                logging.info("Opening DriveuDatabase from file: {}"
                            .format(self.file_path))
                with open(self.file_path, "r") as fp:
                    images = json.load(fp)
            elif label_file_extension == ".yml":
                logging.exception("Yaml support is deprecated. Either use the new .json label files (from download URL received after registration) or checkout <git checkout v1.0> to parse yaml")
                sys.exit(1)
                return False
            else:
                logging.exception("Label file with extension {} not supported. Please use json!".format(label_file_extension))
                sys.exit(1)
                return False
        else:
            logging.exception(
                "Opening DriveuDatabase from File: {} "
                "failed. File or Path incorrect.".format(self.file_path)
            )
            sys.exit(1)
            return False

        for image_dict in images["images"]:
            # parse and store image
            image = DriveuImage()
            image.parse_image_dict(image_dict, data_base_dir)
            self.images.append(image)

        return True


if __name__ == "__main__":
    database = DriveuDatabase("/mnt/data/TLS/DTLD/DTLD_Labels_v2.0/v2.0/Fulda.json")
    if not database.open("/mnt/data/TLS/DTLD"):
        raise "Error opening database path !!!"

    # Visualize image by image
    for idx_d, img in enumerate(database.images):

        img_color = img.get_labeled_image(with_attr=True)
        if img_color.any():
            print(idx_d, "---- num_obj = ", len(img.objects))

            cv2.imshow("rgb", img_color)
            if cv2.waitKey(0) in [ord("q"), 27]:
                break
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import cv2
import numpy as np

__all__ = ["vis", "vis_tensor_targets", "vis_attr"]


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 -9),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])-9),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]-10), font, 0.4, txt_color, thickness=1)

    return img


_COLORS = np.array(
    [
        0.1, 0.2, 1,
        # 0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)


def vis_tensor_targets(input, targets, save_path):
    # visualize targets on input tensor for debug, add by zhanghao
    img1 = np.uint8(input[0].permute(1,2,0).cpu().numpy())
    img1 = img1.astype(np.uint8).copy()

    for ttt in targets[0]:
        if ttt[4] + ttt[3] > 0:
            lt = (int(ttt[1] - ttt[3]/2.0), int(ttt[2] - ttt[4]/2.0))
            rb = (int(ttt[1] + ttt[3]/2.0), int(ttt[2] + ttt[4]/2.0))
            img1 = cv2.rectangle(img1, lt, rb, (0,255,0),1)

            if len(ttt) > 5:
                attr_str = "".join(ttt[5:].int().cpu().numpy().astype(np.str))
                cv2.putText(img1, attr_str, lt, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255), 1)

            cv2.imwrite(save_path, img1)


# colors = [tuple(map(int, np.random.randint(0, high=150, size=(3,)))) for _ in range(100)]
def vis_attr(img, output, bboxes, attr_dict):
    attr_names_list = list(attr_dict.values())
    attr_nums = len(attr_dict)
    for idx, (out, bbox) in enumerate(zip(output, bboxes)):
        x0 = int(bbox[0])
        y0 = int(bbox[1])
        x1 = int(bbox[2])
        y1 = int(bbox[3])

        color = (_COLORS[idx] * 255 * 1.5).astype(np.uint8).tolist()
        # color = colors[idx]
        cv2.rectangle(img, (x0, y1+1), (x0+80, y1+110), color, 2)

        for attr_idx, attr_class_id in enumerate(out[-attr_nums:].int().cpu().numpy()):
            attr_str = attr_names_list[attr_idx][attr_class_id]
            cv2.putText(img, attr_str, (x0, y1+13*attr_idx+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

"""
Eval on local predict txt on disk.

pred:
    label_name, conf, x1, y1, x2, y2

gt:
    yolo-format
"""

import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
# from tabulate import tabulate

from yolox.evaluators.metrics import ConfusionMatrix, ap_per_class, box_iou, AttributeEval, get_table


# class_names_dict = {
#     0 : 'traffic_light'
# }


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def load_yolo_label(txt_path, im_shape=(1024,2048)):
    if not os.path.exists(txt_path):
        return np.empty((0,5))

    height, width = im_shape
    with open(txt_path,"r") as f:
        labels = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
        if len(labels) > 0:
            labels[:, 1:] *= np.array((width, height, width, height))
            labels[:, 1:] = xywh2xyxy(labels[:, 1:])
            return labels
        else:
            return np.empty((0,5))


def load_preds(txt_path, score_thresh=0.0):
    if not os.path.exists(txt_path):
        return np.empty((0,6))

    with open(txt_path,"r") as f:
        preds = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
        mask = preds[:,1] > score_thresh
        return preds[mask][:, [2,3,4,5,1,0]]


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:5], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


# def get_metrics(pred_folder, gt_folder, save_path, img_shape=(1024,2048), nc=1, device="cuda"):
def get_metrics(predictions, 
                groundtruths, 
                save_path, 
                img_shape=(1024,2048), 
                nc=1, 
                device="cuda", 
                with_attr=False,
                attr_num = 8,
                class_names_dict = {0 : 'traffic_light'},
                val_info=""):
    """
    Args:
        predictions : prediction folder(str) or pred-json-dict(dict).
        groundtruths : groundtruths folder(str) or gt-json-dict(dict).

    Returns:
        ap50_95 (float) : COCO style AP of IoU=50:95
        ap50 (float) : VOC 2007 metric AP of IoU=50
        summary (sr): summary info of evaluation.
    """
    os.makedirs(save_path, exist_ok=True)

    pd_ndim = (6+attr_num) if with_attr else 6
    gt_ndim = (5+attr_num) if with_attr else 5

    if isinstance(groundtruths, str):
        gt_files = os.listdir(groundtruths)
    else:
        gt_files = list(groundtruths.keys())

    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    confusion_matrix = ConfusionMatrix(nc=nc)
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    stats, ap, ap_class = [], [], []
    seen = 0

    if with_attr:
        attribute_eval = AttributeEval(attr_num)

    for gt_filename in tqdm(gt_files):
        if isinstance(groundtruths, str):
            gt_labels = load_yolo_label(os.path.join(groundtruths, gt_filename), img_shape)
        else:
            gt_labels = groundtruths.get(gt_filename)
            if gt_labels is None:
                gt_labels = np.empty((0,gt_ndim))

        if isinstance(predictions, str):
            preds = load_preds(os.path.join(predictions, gt_filename))
        else:
            preds = predictions.get(gt_filename)
            if preds is None:
                preds = np.empty((0,pd_ndim))

        gt_labels = torch.from_numpy(gt_labels).to(device)
        preds = torch.from_numpy(preds).to(device)

        # Metrics
        nl, npr = gt_labels.shape[0], preds.shape[0]  # number of labels, predictions
        correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
        seen += 1

        if npr == 0:
            if nl:
                stats.append((correct, *torch.zeros((2, 0), device=device), gt_labels[:, 0]))
                # stats.append((correct, *torch.zeros((3, 0), device=device)))
            continue

        # Evaluate
        if nl:
            correct = process_batch(preds, gt_labels, iouv)
            confusion_matrix.process_batch(preds, gt_labels)
            if with_attr:
                attribute_eval.process_batch(preds, gt_labels)

        stats.append((correct, preds[:, 4], preds[:, 5], gt_labels[:, 0]))  # (correct, conf, pcls, tcls)

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=True, save_dir=save_path, names=class_names_dict)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class
    else:
        mp, mr, map50, map, f1, tp, fp = 0,0,0,0,0,0,0
        nt = torch.zeros(1)

    # Print results
    header = ["images", "labels", "Precision", "Recall", "mAP@.5", "mAP@.5:.95", "f1(best)", "tp(best-f1)", "fp(best-f1)"]   
    infos = [(seen, nt.sum(), mp, mr, map50, map, f1, tp, fp)]
    map_str = get_table(header, infos)
    
    if with_attr:
        attr_str = attribute_eval.get_results()
        map_str = map_str + "\n" + attr_str
    print(map_str)

    with open(save_path + "/res.txt", "a") as ttt:
        ttt.write(val_info + "\n")
        ttt.write(map_str + "\n")

    # Print results per class
    if nc < 50 and nc > 1 and len(stats):
        exp_table = []
        for i, c in enumerate(ap_class):
            exp_table.append((class_names_dict[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        class_header = ["class_name", "images", "labels", "P", "R", "mAP@.5", "mAP@.5:.95"]   
        map_str_class = get_table(class_header, exp_table)
        print(map_str_class)
        with open(save_path + "/map_perclass.txt", "a") as ttt:
            ttt.write(map_str_class + "\n")

    confusion_matrix.plot(save_dir=save_path, names=list(class_names_dict.values()))

    if with_attr:
        return map, map50, map_str, attribute_eval.get_accurate()
    else:
        return map, map50, map_str


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default="/mnt/Perception/zhanghao/DTLD_Yolo/labels/val", help='gt-yolo txt path')
    parser.add_argument('--pd', type=str, default="/home/zhanghao/code/master/2_DET2D/yolov5/runs/val/exp/labels", help='pred txt path')
    parser.add_argument('--save_path', type=str, default="./runs/local_eval", help='result save path')
    opt = parser.parse_args()
    print(vars(opt))
    return opt


if __name__ == "__main__":
    opt = parse_opt()

    get_metrics(
        opt.pd,
        opt.gt,
        opt.save_path,
        val_info=str(vars(opt))
    )




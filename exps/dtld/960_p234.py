# encoding: utf-8
import os

import torch
import torch.distributed as dist

from yolox.data import get_yolox_datadir
from yolox.exp import Exp as MyExp

"""
ATTENTION:
    Changed input-size, must change cache !!!

"""

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.output_dir = "/home/zhanghao/code/master/2_DET2D/YOLOX/YOLOX_outputs/"
        
        self.num_classes = 1
        self.depth = 0.33
        self.width = 0.50
        

        self.input_size = (640, 640)  # (height, width)
        self.test_size  = (640, 640)
        # self.input_size = (480, 960)  # (height, width)
        # self.test_size  = (480, 960)

        self.data_num_workers = 4

        # --------------- transform config ----------------- #
        # prob of applying mosaic aug
        self.mosaic_prob = 0.0
        # prob of applying mixup aug
        self.mixup_prob = 0.0
        # prob of applying hsv aug
        self.hsv_prob = 0.1
        # prob of applying flip aug
        self.flip_prob = 0.5
        # rotation angle range, for example, if set to 2, the true range is (-2, 2)
        self.degrees = 4.0
        # translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
        self.translate = 0.05
        self.mosaic_scale = (0.9, 1.1)
        # apply mixup aug or not
        self.enable_mixup = False
        self.mixup_scale = (0.7, 1.3)
        # shear angle range, for example, if set to 2, the true range is (-2, 2)
        self.shear = 0.1

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # add by zhanghao
        self.eval_interval = 1
        self.save_history_ckpt = True
        self.save_interval = 5

        # --------------  training config --------------------- #
        self.warmup_epochs = 2
        # max training epoch
        self.max_epoch = 100
        # minimum learning rate during warmup
        self.warmup_lr = 0.0
        self.min_lr_ratio = 0.25
        # learning rate for one image. During training, lr will multiply batchsize.
        self.basic_lr_per_img = 0.01 / 64.0
        # name of LRScheduler
        self.scheduler = "yoloxwarmcos"
        # last #epoch to close augmention like mosaic
        self.no_aug_epochs = 10

        # weight decay of optimizer
        self.weight_decay = 5e-4
        # momentum of optimizer
        self.momentum = 0.9
        # log period in iter, for example,
        # if set to 1, user could see log every iteration.
        self.print_interval = 10


    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import (
            DTLDDetection,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import (
            wait_for_the_master,
            get_local_rank,
        )
        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            dataset = DTLDDetection(
                data_dir="/data/DTLD",
                json_path="/data/DTLD/v2.0/DTLD_train.json",
                dataset_name="DTLD_train",
#                 json_path="/data/DTLD/v2.0/Fulda.json",
#                 dataset_name="Fulda",
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
                cache=cache_img,
            )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
                with_dtld_attr=True),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import DTLDDetection, ValTransform
        valdataset = DTLDDetection(
            data_dir="/data/DTLD",
            json_path="/data/DTLD/v2.0/DTLD_test.json",
            dataset_name="DTLD_test",
#             json_path="/data/DTLD/v2.0/Fulda.json",
#             dataset_name="Fulda",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
            cache=True
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import DTLDEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = DTLDEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            output_path=self.output_dir+"/" + self.exp_name + "/eval/",
            with_attr=True
        )
        return evaluator


    def get_model(self):
        from torch import nn
        from yolox.models import YOLOX, YOLOPAFPN, YOLOPAFPN_234, YOLOXHead, YOLOXHeadAttr
        from yolox.data.datasets import DTLD_ATTRIBUTES

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [128, 256, 512, 1024]
            # backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            backbone = YOLOPAFPN_234(self.depth, self.width, in_channels=in_channels, act=self.act)
            # head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            head = YOLOXHeadAttr(self.num_classes, self.width, in_channels=in_channels, act=self.act, 
                                 attribute_dict=DTLD_ATTRIBUTES,
                                 attr_weight=1.0)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model


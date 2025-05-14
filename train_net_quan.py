#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from ubteacher import add_ubteacher_config
from ubteacher.engine.trainer import UBTeacherTrainer, BaselineTrainer

# hacky way to register
from ubteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN
from ubteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from ubteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import ubteacher.data.datasets.builtin

from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel

import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

# 注册数据集

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
import pycocotools
#声明类别，尽量保持
CLASS_NAMES =["1"]
# 数据集路径
DATASET_ROOT = '/mnt/sdd/zhanghexiang/GroupRCNN/data/coco'
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations_unlabel')

TRAIN_PATH = os.path.join(DATASET_ROOT, 'images')
VAL_PATH = os.path.join(DATASET_ROOT, 'images')

TRAIN_JSON = '/mnt/sdd/zhanghexiang/GroupRCNN/data/coco/annotations_unlabel/instances_train2014.json'
#VAL_JSON = os.path.join(ANN_ROOT, 'val.json')
VAL_JSON = '/mnt/sdd/zhanghexiang/GroupRCNN/data/coco/annotations_unlabel/instances_val2014.json'

# 声明数据集的子集
PREDEFINED_SPLITS_DATASET = {
    "coco_zhx_train": (TRAIN_PATH, TRAIN_JSON),
    "coco_zhx_val": (VAL_PATH, VAL_JSON),
}
#===========以下有两种注册数据集的方法，本人直接用的第二个plain_register_dataset的方式 也可以用register_dataset的形式==================
#注册数据集（这一步就是将自定义数据集注册进Detectron2）
'''
def register_dataset():
    """
    purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    """
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        register_dataset_instances(name=key,
                                   json_file=json_file,
                                   image_root=image_root)


#注册数据集实例，加载数据集中的对象实例
def register_dataset_instances(name, json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file,
                                  image_root=image_root,
                                  evaluator_type="coco")
'''
#=============================
# 注册数据集和元数据
def plain_register_dataset():
    #训练集
    DatasetCatalog.register("coco_zhx_train", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH))
    MetadataCatalog.get("coco_zhx_train").set(thing_classes=CLASS_NAMES,  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
                                                    evaluator_type='coco', # 指定评估方式
                                                    json_file=TRAIN_JSON,
                                                    image_root=TRAIN_PATH)

    #DatasetCatalog.register("coco_my_val", lambda: load_coco_json(VAL_JSON, VAL_PATH, "coco_2017_val"))
    #验证/测试集
    DatasetCatalog.register("coco_zhx_val", lambda: load_coco_json(VAL_JSON, VAL_PATH))
    MetadataCatalog.get("coco_zhx_val").set(thing_classes=CLASS_NAMES, # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
                                                evaluator_type='coco', # 指定评估方式
                                                json_file=VAL_JSON,
                                                image_root=VAL_PATH)
# 查看数据集标注，可视化检查数据集标注是否正确，
#这个也可以自己写脚本判断，其实就是判断标注框是否超越图像边界
#可选择使用此方法
# def checkout_dataset_annotation(name="coco_my_val"):
#     #dataset_dicts = load_coco_json(TRAIN_JSON, TRAIN_PATH, name)
#     dataset_dicts = load_coco_json(TRAIN_JSON, TRAIN_PATH)
#     print(len(dataset_dicts))
#     for i, d in enumerate(dataset_dicts,0):
#         #print(d)
#         img = cv2.imread(d["file_name"])
#         visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(name), scale=1.5)
#         vis = visualizer.draw_dataset_dict(d)
#         #cv2.imshow('show', vis.get_image()[:, :, ::-1])
#         cv2.imwrite('out/'+str(i) + '.jpg',vis.get_image()[:, :, ::-1])
#         #cv2.waitKey(0)
#         # if i == 200:
#         #     break
        
# class Trainer(DefaultTrainer):
#     """
#     We use the "DefaultTrainer" which contains pre-defined default logic for
#     standard training workflow. They may not work for you, especially if you
#     are working on a new research project. In that case you can write your
#     own training loop. You can use "tools/plain_train_net.py" as an example.
#     """

#     @classmethod
#     def build_evaluator(cls, cfg, dataset_name, output_folder=None):
#         """
#         Create evaluator(s) for a given dataset.
#         This uses the special metadata "evaluator_type" associated with each builtin dataset.
#         For your own dataset, you can simply create an evaluator manually in your
#         script and do not have to worry about the hacky if-else logic here.
#         """
#         if output_folder is None:
#             output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
#         evaluator_list = []
#         evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
#         if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
#             evaluator_list.append(
#                 SemSegEvaluator(
#                     dataset_name,
#                     distributed=True,
#                     output_dir=output_folder,
#                 )
#             )
#         if evaluator_type in ["coco", "coco_panoptic_seg"]:
#             evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
#         if evaluator_type == "coco_panoptic_seg":
#             evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
#         if evaluator_type == "cityscapes_instance":
#             assert (
#                 torch.cuda.device_count() >= comm.get_rank()
#             ), "CityscapesEvaluator currently do not work with multiple machines."
#             return CityscapesInstanceEvaluator(dataset_name)
#         if evaluator_type == "cityscapes_sem_seg":
#             assert (
#                 torch.cuda.device_count() >= comm.get_rank()
#             ), "CityscapesEvaluator currently do not work with multiple machines."
#             return CityscapesSemSegEvaluator(dataset_name)
#         elif evaluator_type == "pascal_voc":
#             return PascalVOCDetectionEvaluator(dataset_name)
#         elif evaluator_type == "lvis":
#             return LVISEvaluator(dataset_name, output_dir=output_folder)
#         if len(evaluator_list) == 0:
#             raise NotImplementedError(
#                 "no Evaluator for the dataset {} with the type {}".format(
#                     dataset_name, evaluator_type
#                 )
#             )
#         elif len(evaluator_list) == 1:
#             return evaluator_list[0]
#         return DatasetEvaluators(evaluator_list)

#     @classmethod
#     def test_with_TTA(cls, cfg, model):
#         logger = logging.getLogger("detectron2.trainer")
#         # In the end of training, run an evaluation with TTA
#         # Only support some R-CNN models.
#         logger.info("Running inference with test-time augmentation ...")
#         model = GeneralizedRCNNWithTTA(cfg, model)
#         evaluators = [
#             cls.build_evaluator(
#                 cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
#             )
#             for name in cfg.DATASETS.TEST
#         ]
#         res = cls.test(cfg, model, evaluators)
#         res = OrderedDict({k + "_TTA": v for k, v in res.items()})
#         return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.DATASETS.TRAIN = ("coco_zhx_train",) # 训练数据集名称，修改
    cfg.DATASETS.TEST = ("coco_zhx_val",) # 训练数据集名称，修改
    cfg.MODEL.RETINANET.NUM_CLASSES = 1 # 修改自己的类别数

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    plain_register_dataset() #  # 修改
    if cfg.SEMISUPNET.Trainer == "ubteacher":
        Trainer = UBTeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "baseline":
        Trainer = BaselineTrainer
    else:
        raise ValueError("Trainer Name is not found.")

    if args.eval_only:
        if cfg.SEMISUPNET.Trainer == "ubteacher":
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            res = Trainer.test(cfg, ensem_ts_model.modelTeacher)



    else:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)

        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )



#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from missod import add_ubteacher_config
from missod.engine.trainer import UBTeacherTrainer, BaselineTrainer

# hacky way to register
from missod.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN
from missod.modeling.proposal_generator.rpn import PseudoLabRPN
from missod.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab

from MISSOD.missod.modeling.mhfpn import build_resnet_mhfpn2_backbone
import missod.data.datasets.builtin

from missod.modeling.meta_arch.ts_ensemble import EnsembleTSModel

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
import pycocotools
from detectron2.engine import HookBase


CLASS_NAMES =["lesion"]
DATASET_ROOT = '/mnt/sdd/xxxxxx/coco/'
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations/')
TRAIN_PATH = os.path.join(DATASET_ROOT, 'images')
VAL_PATH = os.path.join(DATASET_ROOT, 'images')

TRAIN_LABEL_JSON =  os.path.join(ANN_ROOT, 'instances_train2014.json')
TRAIN_UNLABEL_JSON = os.path.join(ANN_ROOT, 'instances_train2014_unlabel.json')
VAL_JSON = os.path.join(ANN_ROOT, 'instances_val2014.json')


PREDEFINED_SPLITS_DATASET = {
    "coco_xxx_train_label": (TRAIN_PATH, TRAIN_LABEL_JSON),
    "coco_xxx_train_unlabel": (TRAIN_PATH, TRAIN_UNLABEL_JSON),
    "coco_xxx_val": (VAL_PATH, VAL_JSON),
}

#=============================
def plain_register_dataset():
    #train 
    DatasetCatalog.register("coco_xxx_train_label", lambda: load_coco_json(TRAIN_LABEL_JSON, TRAIN_PATH))
    MetadataCatalog.get("coco_xxx_train_label").set(thing_classes=CLASS_NAMES,  
                                                    evaluator_type='coco', 
                                                    json_file=TRAIN_LABEL_JSON,
                                                    image_root=TRAIN_PATH)
    DatasetCatalog.register("coco_xxx_train_unlabel", lambda: load_coco_json(TRAIN_UNLABEL_JSON, TRAIN_PATH))
    MetadataCatalog.get("coco_xxx_train_unlabel").set(thing_classes=CLASS_NAMES,  
                                                    evaluator_type='coco', 
                                                    json_file=TRAIN_UNLABEL_JSON,
                                                    image_root=TRAIN_PATH)                                                

    #val
    DatasetCatalog.register("coco_xxx_val", lambda: load_coco_json(VAL_JSON, VAL_PATH))
    MetadataCatalog.get("coco_xxx_val").set(thing_classes=CLASS_NAMES, 
                                                evaluator_type='coco',
                                                json_file=VAL_JSON,
                                                image_root=VAL_PATH)

class BestCheckpointer(HookBase):
  def __init__(self):
      super().__init__()

  def after_step(self):
    # No way to use **kwargs
    ##ONly do this analys when trainer.iter is divisle by checkpoint_epochs
    curr_val = self.trainer.storage.latest().get('bbox/AP50', 0)
    import math
    if type(curr_val) != int:
        curr_val = curr_val[0]
        if math.isnan(curr_val):
            curr_val = 0
    try:
        _ = self.trainer.storage.history('max_bbox/AP50')
    except:
        self.trainer.storage.put_scalar('max_bbox/AP50', curr_val)

    max_val = self.trainer.storage.history('max_bbox/AP50')._data[-1][0]
    if curr_val > max_val:
        print("\n%s > %s\n"%(curr_val,max_val))
        self.trainer.storage.put_scalar('max_bbox/AP50', curr_val)
        self.trainer.checkpointer.save("model_best_{}".format(float(curr_val)))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    plain_register_dataset()
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

    trainer = Trainer(cfg)
    trainer.register_hooks([BestCheckpointer()])
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

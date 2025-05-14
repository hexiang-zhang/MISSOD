# 导入必要的库
import torch,torchvision
import cv2
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
import torchvision
from ubteacher.engine.trainer import UBTeacherTrainer
from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog
from ubteacher.modeling.mhfpn2 import build_resnet_mhfpn2_backbone
from detectron2.utils.visualizer import ColorMode, Visualizer
from pycocotools.coco import COCO



def bbox_iou(box1, box2):
    """
    计算两个框之间的IOU值
    Args:
        box1: 第一个框，四个坐标，形状为 [4] 的int型列表
        box2: 第二个框，四个坐标，形状为 [4] 的int型列表
    Returns:
        iou: IOU值，float型
    """
    # 将两个框的坐标转换为左上角和右下角的坐标形式
    box1_tl = torch.Tensor(box1[:2]).int()
    box1_br = torch.Tensor(box1[2:]).int()
    box2_tl = torch.Tensor(box2[:2]).int()
    box2_br = torch.Tensor(box2[2:]).int()

    # 计算两个框的面积
    box1_area = (box1_br[0] - box1_tl[0]) * (box1_br[1] - box1_tl[1])
    box2_area = (box2_br[0] - box2_tl[0]) * (box2_br[1] - box2_tl[1])

    # 计算两个框相交部分的左上角和右下角坐标
    inter_tl = torch.max(box1_tl, box2_tl)
    inter_br = torch.min(box1_br, box2_br)

    # 计算相交部分的面积
    inter_area = torch.clamp(inter_br[0] - inter_tl[0], min=0) * torch.clamp(inter_br[1] - inter_tl[1], min=0)

    # 计算IOU值
    iou = inter_area.float() / (box1_area + box2_area - inter_area).float()

    return iou.item()

class CustomPredictor:

    def __init__(self, cfg):
        self.cfg = cfg.clone()

        Trainer = UBTeacherTrainer

        model = Trainer.build_model(cfg)
        model_teacher = Trainer.build_model(cfg)
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        DetectionCheckpointer(
            ensem_ts_model,
            save_dir=cfg.OUTPUT_DIR
        ).resume_or_load(cfg.MODEL.WEIGHTS, resume=True)

        model = ensem_ts_model.modelTeacher
        self.model = model

        self.model.eval()

    def __call__(self, original_image):
        with torch.no_grad():
            original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = original_image
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions


# 定义模型结构
cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu'
cfg.merge_from_file("/mnt/sdd/zhanghexiang/unbiased-teacher/configs/Base-RCNN-FPN.yaml")
cfg.MODEL.WEIGHTS = "/mnt/sdd/zhanghexiang/unbiased-teacher/output_mhfpn_full/model_best_35.82365476855631.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80

# # 定义预测器
predictor = CustomPredictor(cfg)
register_coco_instances("my_dataset_val", {}, "/mnt/sdd/zhanghexiang/coco_deep/annotations/instances_val2014.json", "/mnt/sdd/zhanghexiang/coco_deep/images")
f = open('output.txt', 'w')
# 加载元数据
metadata = MetadataCatalog.get("my_dataset_val")
metadata.thing_colors = [(255, 0, 0)]
dataset_dicts = DatasetCatalog.get("my_dataset_val")

ann_file = '/mnt/sdd/zhanghexiang/deep.json'
dataset_dir = "/mnt/sdd/zhanghexiang/coco_deep/images/"

coco = COCO(ann_file)
catIds = coco.getCatIds(catNms=['1']) # catIds=1 表示人这一类
imgIds = coco.getImgIds(catIds=catIds ) # 图片id，许多值
for i in range(len(imgIds)):
    img = coco.loadImgs(imgIds[i])[0]
    
    image = cv2.imread(dataset_dir + img['file_name'])
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    annos = coco.loadAnns(annIds)

    bbox = annos[0]['bbox']
    x, y, w, h = bbox
    bbox_gt = [x,y,x+w,y+h]
    outputs = predictor(image)
    
    # NMS
    instances = outputs["instances"]
    scores = instances.scores
    keep = torchvision.ops.nms(instances.pred_boxes.tensor, scores, iou_threshold=0.5)
    instances = instances[keep]
    if len(keep)>0:
        # 对分数进行排序，并获取排序后的索引
        sorted_idx = torch.argsort(scores, descending=True)

        # 筛选出分数最高的预测框
        keep = [int(sorted_idx[0])]
        # 将keep变为list格式，以符合detectron2的要求
        instances = instances[torch.tensor(keep)]
        # 将Boxes对象转换为列表形式
        box_list = instances._fields['pred_boxes'].tensor.tolist()

        # 遍历列表，将每个元素转换为整型
        iou = bbox_iou(box1=bbox_gt,box2=box_list[0])

        print('{} iou is {}'.format(img['file_name'],iou), file=f)
    
    else:
        print("{} is none".format(img['file_name']), file=f)

           
    # 可视化
    v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0, instance_mode=ColorMode.SEGMENTATION)
    out = v.draw_instance_predictions(instances.to("cpu"))
    cv2.imwrite("/mnt/sdd/zhanghexiang/vis/zhx/{}".format(img['file_name']), out.get_image()[:, :, ::-1])
    last_img = cv2.imread("/mnt/sdd/zhanghexiang/vis/zhx/{}".format(img['file_name']))
    anno_image = cv2.rectangle(last_img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
    cv2.imwrite("/mnt/sdd/zhanghexiang/vis/test/{}".format(img['file_name']), anno_image)

f.close()



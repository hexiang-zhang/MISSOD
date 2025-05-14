import cv2
import numpy as np
import torch
from torch.nn import functional as F
from detectron2.config import get_cfg
from ubteacher.engine.trainer import UBTeacherTrainer
from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from detectron2.checkpoint import DetectionCheckpointer
from ubteacher.modeling.cefpn import build_resnet_cefpn_backbone
from ubteacher.modeling.mhfpn2 import build_resnet_mhfpn2_backbone
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from detectron2.data.detection_utils import read_image
from scipy.ndimage import gaussian_filter

     
# 加载配置文件和模型
cfg = get_cfg()
cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu'
cfg.merge_from_file("/mnt/sdd/zhanghexiang/unbiased-teacher/configs/Base-RCNN-MHFPN.yaml")
cfg.MODEL.WEIGHTS = "/mnt/sdd/zhanghexiang/unbiased-teacher/ex/okk_3/3-brats/output_mhfpn_full_0.890_7_1/model_best_88.20081838132154.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80

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

def smooth_matrix(matrix, sigma=22):
    """
    对输入的numpy矩阵进行平滑处理。

    参数:
        matrix (np.array): 一个100x100的numpy矩阵。
        sigma (float, 可选): 高斯滤波器的标准差，默认为1。

    返回:
        np.array: 平滑后的矩阵。
    """
    smoothed_matrix = gaussian_filter(matrix, sigma)
    return smoothed_matrix

# 定义数据
image_path = '/mnt/sdd/zhanghexiang/coco_brats/images/51.png'
model = CustomPredictor(cfg)


# 定义SmoothGrad函数
def draw_heatmap(image, model):
    with torch.no_grad():
        outputs = model(image)
        # cls_logits = outputs["instances"].pred_logits.softmax(-1)
        cls_scores = outputs["instances"].scores.cpu().numpy()
        boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()

        heatmap = np.zeros_like(image[:, :, 0],dtype=np.float)
        for score, box in zip(cls_scores, boxes):
            x1, y1, x2, y2 = box.astype(np.int32)
            heatmap[y1:y2, x1:x2] += score

        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        heatmap = smooth_matrix(heatmap)
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.imshow(heatmap, cmap='jet', interpolation='bilinear', alpha=0.5)
        plt.axis('off')
        plt.savefig('2.png', transparent=True, dpi=100, pad_inches = 0)


class_id = 0  # 根据需要更改目标类别ID
# 读取图像并预处理
image = cv2.imread(image_path)
draw_heatmap(image, model)


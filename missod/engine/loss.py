import torch
from torch import nn
import numpy as np
import math
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
import torch


class pseudo_loss(nn.Module):
    def __init__(
        self,
        all_unlabel_data,
        pesudo_instances,
        cur_threshold = 0.7,
        logistic_k = 5,
        max_iter = 10000,
        now_iter = 1,
    ):
        super(pseudo_loss, self)    
        self.strong_data = all_unlabel_data
        self.pesudo_instances = pesudo_instances
        self.cur_threshold = cur_threshold
        self.logistic_k = logistic_k
        self.box_reg_loss_type = 'iou'
        self.max_iter = max_iter
        self.now_iter = now_iter

    def losses(self):
        return {
            "medical_loss_box_reg_iou_pesudo": self.box_reg_loss(),
        }
    def iou_loss(self,pred_boxes, gt_boxes):
        """
        Compute IOU between two sets of bounding boxes.
        Args:
            pred_boxes: Tensor[N, 4]
            gt_boxes: Tensor[M, 4]
        Returns:
            iou: Tensor[N, M]
        """
        n = pred_boxes.size(0)
        m = gt_boxes.size(0)
        if n > 0:
            pred_boxes = pred_boxes.unsqueeze(1).expand(n, m, 4)
            gt_boxes = gt_boxes.unsqueeze(0).expand(n, m, 4)

            # Compute the left-top and right-bottom points of the intersection
            lt = torch.max(pred_boxes[..., :2], gt_boxes[..., :2])
            rb = torch.min(pred_boxes[..., 2:], gt_boxes[..., 2:])

            # Compute the width and height of the intersection
            wh = (rb - lt + 1).clamp(min=0)
            inter = wh[..., 0] * wh[..., 1]

            # Compute the area of the prediction box and the ground truth box
            area_pred = (pred_boxes[..., 2] - pred_boxes[..., 0] + 1) * (pred_boxes[..., 3] - pred_boxes[..., 1] + 1)
            area_gt = (gt_boxes[..., 2] - gt_boxes[..., 0] + 1) * (gt_boxes[..., 3] - gt_boxes[..., 1] + 1)

            # Compute the IOU between the prediction box and the ground truth box
            iou = inter / (area_pred + area_gt - inter)
            max_iou, max_indices = iou.max(dim=1)
            iou_loss = (1 - max_iou).sum()

        else:
            iou = torch.zeros(gt_boxes.size(0))
            iou_loss = (1 - iou).sum()
        return iou_loss.cuda()

      
    def my_funcv2(self,x,cur_threshold):
        return 1 - (x - cur_threshold)/(1-cur_threshold)

    def train_loss(self,k=15,x0=0.5):
        x = torch.tensor(self.now_iter) / torch.tensor(self.max_iter)
        y = 1 / (1 + torch.exp(-k * (x - x0)))
        return y

    def box_reg_loss(self):
        assert len(self.strong_data) == len(self.pesudo_instances),'some thing wrong in the data in'
        total_loss = 0
        for idx in range(len(self.strong_data)):
            data_ins = self.strong_data[idx]['instances']
            pre_ins = self.pesudo_instances[idx]
            data_scores = data_ins.scores 
            pre_scores = pre_ins.scores

            data_box = data_ins.gt_boxes.tensor
            pre_box = pre_ins.gt_boxes.tensor
            if data_scores.shape == torch.Size([0]):
                total_loss = total_loss + self.train_loss().cuda()
            else:     
                loss_weight = self.my_funcv2(data_scores,self.cur_threshold).mean().cuda()
                iou_loss = self.iou_loss(pre_box,data_box)
                total_loss += torch.sum(loss_weight * iou_loss)

        
        return total_loss    



            
             


    

        
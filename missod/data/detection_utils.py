import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFilter
import random
import torch
import logging
from torchvision.transforms import functional as F

from detectron2.data.transforms.augmentation import Augmentation
from detectron2.data.transforms.augmentation_impl import NoOpTransform
import numpy as np

class UnsharpMask:

    def __init__(self, radius=2, percent=150, threshold=3):
        self.radius = radius
        self.percent = percent
        self.threshold = threshold

    def __call__(self, x):
        return x.filter(ImageFilter.UnsharpMask(self.radius, self.percent, self.threshold))

def erase_square_rotation(img, i: int, j: int, h: int, w: int):
    p = random.random()
    if p > 0.5:
        width = w
    else:
        width = h    

    angle_list = [0,1,2,3]
    roll_list = []
    image = F.to_pil_image(img,)
    item_num = int(width / 20)

    box_list = []
    for r in range(0,item_num):
        for l in range(0,item_num):
            box = (i+l*20, j+r*20, i+(l+1)*20, j+(r+1)*20)
            box_list.append(box)
    image_list = [image.crop(box) for box in box_list]
    for i in range(len(image_list)):
        v = random.choice(angle_list)
        roll_list.append(image_list[i].rotate(90*v))
    for i, box in enumerate(box_list):
        image.paste(roll_list[i],box)
    for i in range(int(float(len(box_list))*0.3)):
        x = random.randint(1,len(box_list)) - 1
        image.paste((0,0,0),box_list[x])

    image = F.pil_to_tensor(image)

    return image


def erase_square_rotation(img, i: int, j: int, h: int, w: int):
    """ Erase the input Tensor Image with given value.
    This transform does not support PIL Image.

    Args:
        img (PIL Image): PIL image of size (C, H, W) to be erased
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the erased region.
        w (int): Width of the erased region.
        inplace(bool, optional): For in-place operations. By default is set False.

    Returns:
        Tensor Image: Erased image.
    """
    p = random.random()
    if p > 0.5:
        h = w
    else:
        w = h    
    if h != w:
        raise TypeError('h should be equal to the w. Now h is {}, w is {}'.format(int(h),int(w)))

    angle_list = [0,1,2,3,4]
    v = random.choice(angle_list)
    #trans_pil = transforms.ToPILImage()
    image = F.to_pil_image(img,)
    box=(i,j,i+h,j+w)
    roi = image.crop(box)
    roi = roi.rotate(90*v)
    image.paste(roi,box)
    image = F.pil_to_tensor(image)

    return image

class RandomRollErasing(transforms.RandomErasing):

    def forward(self, img):
        """
        Args:
            img (Tensor): Tensor image to be erased.

        Returns:
            img (Tensor): Erased Tensor image.
        """
        if torch.rand(1) < self.p:

            # cast self.value to script acceptable type
            if isinstance(self.value, (int, float)):
                value = [self.value, ]
            elif isinstance(self.value, str):
                value = None
            elif isinstance(self.value, tuple):
                value = list(self.value)
            else:
                value = self.value

            if value is not None and not (len(value) in (1, img.shape[-3])):
                raise ValueError(
                    "If value is a sequence, it should have either a single value or "
                    "{} (number of input channels)".format(img.shape[-3])
                )

            x, y, h, w, _ = self.get_params(img, scale=self.scale, ratio=self.ratio, value=value)
            return erase_square_rotation(img, x, y, h, w)
        return img

def build_strong_augmentation(cfg, is_train):
    """
    Create a list of :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """

    logger = logging.getLogger(__name__)
    augmentation = []
    if is_train:
        base_transform = transforms.Compose([
            transforms.RandomApply([UnsharpMask()], p=1.0),
            transforms.RandomGrayscale(p=0.2),
        ])
        strong_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                RandomRollErasing(
                    p=0.8, scale=(0.15, 0.2), ratio=(0.9, 1.1), value="random"
                ),
                RandomRollErasing(
                    p=1.0 , scale=(0.1, 0.15), ratio=(0.9, 1.1), value="random"
                ),
                RandomRollErasing(
                    p=1.0, scale=(0.06, 0.1), ratio=(0.9, 1.1), value="random"
                ),
                transforms.ToPILImage(),
            ]
        )
        augmentation.append(base_transform)
        augmentation.append(strong_transform)
        logger.info("Augmentations used in training: " + str(augmentation))

    return transforms.Compose(augmentation)


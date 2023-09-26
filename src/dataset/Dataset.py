from torch.utils.data import Dataset
import cv2
import os
from os.path import join
import numpy as np
import torch
import torchvision.transforms as transforms
import torchdata.datapipes as dp
import torchvision.transforms.functional as TF
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from PIL import Image

def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

class DistillDataset(dp.iter.IterDataPipe):
    def __init__(self, path_lr, path_ls, poses, yaw) -> None:
        super().__init__()
        self.path_lr = path_lr
        self.path_ls = path_ls
        self.poses = poses
        self.yaw = yaw

    def __iter__(self):
        plrs = os.listdir(self.path_lr)
        plss = os.listdir(self.path_ls)
        sorted(plrs)
        sorted(plss)
        for plr, pls in zip(plrs, plss):
            plr = os.path.join(self.path_lr, plr)
            pls = os.path.join(self.path_ls, pls)
            index = os.path.abspath(plr).split('_')[-1].replace('.png','')
            if index in self.poses:
                x, y, rot = self.poses[index]
                rot = rot - self.yaw / 180 * torch.pi
                pose=[x,y,rot]
                yield plr, pls, pose            

@dp.functional_datapipe('filter_low_feature')
class LowFeatureFilter(dp.iter.IterDataPipe):
    def __init__(self, dp) -> None:
        super().__init__()
        self.dp = dp
        self.orb = cv2.ORB_create()

    def __iter__(self):
        for plr, pls, pose in self.dp:
            image = cv2.imread(plr)
            kp = self.orb.detect(image, None)
            kp, des = self.orb.compute(image, kp)
            if len(kp) > 100:
                yield plr, pls, pose

def load_image(path):
    image = TF.pil_to_tensor(Image.open(path).convert('RGB'))
    image = TF.convert_image_dtype(image, torch.float32)
    image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image

def load_image_pair(datachunk):
    plr, pls, pose = datachunk
    image_lr = load_image(plr)
    image_sr = load_image(pls)
    pose = torch.tensor(pose)
    return image_lr, image_sr, pose, plr, pls
    

def distill_dataset_dp(path_lr, path_sr, poses, yaw,batch_size=1):
    p = DistillDataset(path_lr, path_sr, poses, yaw).sharding_filter().shuffle().filter_low_feature().map(load_image_pair)
    return p
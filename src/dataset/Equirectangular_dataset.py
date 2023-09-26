import numpy as np
import cv2
from torch.utils.data import Dataset
import os
from os.path import join
from feature.Local_extractor import Local_extractor
from PIL import Image

class Equirectangular:
    RADIUS = 128

    def __init__(self, **config):
        self.config = config
        self.pitch_list = np.linspace(-config['slice']['PITCH']['range'] / 2, config['slice']['PITCH']['range'] / 2,
                                      config['slice']['PITCH']['num'])
        self.yaw_list = np.linspace(0, 360, config['slice']['YAW']['num'] + 1)[:-1]
        self.FOV = config['slice']['FOV']
        self.height = config['slice']['SHAPE']['height']
        self.width = config['slice']['SHAPE']['width']

    def GetPerspective(self, image, THETA, PHI):
        # self._img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        [self._height, self._width, _] = image.shape
        equ_h = self._height
        equ_w = self._width
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0
        wFOV = self.FOV
        hFOV = float(self.height) / self.width * wFOV
        c_x = (self.width - 1) / 2.0
        c_y = (self.height - 1) / 2.0

        wangle = (180 - wFOV) / 2.0
        w_len = 2 * self.RADIUS * np.sin(np.radians(wFOV / 2.0)) / np.sin(np.radians(wangle))
        w_interval = w_len / (self.width - 1)

        hangle = (180 - hFOV) / 2.0
        h_len = 2 * self.RADIUS * np.sin(np.radians(hFOV / 2.0)) / np.sin(np.radians(hangle))
        h_interval = h_len / (self.height - 1)
        x_map = np.zeros([self.height, self.width], np.float32) + self.RADIUS
        y_map = np.tile((np.arange(0, self.width) - c_x) * w_interval, [self.height, 1])
        z_map = -np.tile((np.arange(0, self.height) - c_y) * h_interval, [self.width, 1]).T
        D = np.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)
        xyz = np.zeros([self.height, self.width, 3], float)
        xyz[:, :, 0] = (self.RADIUS / D * x_map)[:, :]
        xyz[:, :, 1] = (self.RADIUS / D * y_map)[:, :]
        xyz[:, :, 2] = (self.RADIUS / D * z_map)[:, :]

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))
        xyz = xyz.reshape([self.height * self.width, 3]).T
        xyz = np.dot(R1, xyz)
        xyz = np.dot(R2, xyz).T
        lat = np.arcsin(xyz[:, 2] / self.RADIUS)
        lon = np.zeros([self.height * self.width], float)
        theta = np.arctan(xyz[:, 1] / xyz[:, 0])
        idx1 = xyz[:, 0] > 0
        idx2 = xyz[:, 1] > 0

        idx3 = ((1 - idx1) * idx2).astype(bool)
        idx4 = ((1 - idx1) * (1 - idx2)).astype(bool)

        lon[idx1] = theta[idx1]
        lon[idx3] = theta[idx3] + np.pi
        lon[idx4] = theta[idx4] - np.pi

        lon = lon.reshape([self.height, self.width]) / np.pi * 180
        lat = -lat.reshape([self.height, self.width]) / np.pi * 180
        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90 * equ_cy + equ_cy

        persp = cv2.remap(image, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_WRAP)

        return persp

class Equirectangular_dataset(Dataset):
    def __init__(self, path=None, frame_pose=None, extractors=None,local_feature_type=None, device=None, **config):
        self.config = config
        self.extractors = extractors
        self.equ2pers = Equirectangular(**config)
        local_feature = Local_extractor(local_feature_type, device)
        self.local_feature_extractor = local_feature.extractor()
        self.YAW_num, self.PITCH_num = config['slice']['YAW']['num'], config['slice']['PITCH']['num']
        self.group_size = self.YAW_num * self.PITCH_num
        images_path = sorted(os.listdir(path))
        self.frame_pose,self.images_path=[],[]
        for k, v in frame_pose.items():
            self.frame_pose.append(v)
            self.images_path.append(join(path,images_path[int(k)-1]))
        self.GT_num=len(self.frame_pose)
        self.orb = cv2.ORB_create()

    def __getitem__(self, index):
        image_index = index // self.group_size
        sub_index = index - image_index * self.group_size
        image = cv2.imread(self.images_path[image_index])
        YAW = self.equ2pers.yaw_list[sub_index % self.YAW_num]
        PITCH = self.equ2pers.pitch_list[sub_index // self.YAW_num]
        ### Get descriptor
        descriptor,valid = self.extract_descriptor_equi(image, YAW, PITCH)
        ### Get frame pose
        x, y, rot = self.frame_pose[image_index]
        rot = rot - YAW / 180 * np.pi
        path = self.images_path[image_index] + '+' + str(YAW) + ',' + str(PITCH)
        return path, descriptor, np.array((x, y, rot)),valid

    def __len__(self):
        return self.GT_num * self.group_size

    def process(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        width, height = img.size
        scale = 640 / width
        newsize = (640, int(height * scale))
        img = img.resize(newsize)
        return img
    def extract_descriptor_equi(self, image, YAW, PITCH):
        image = self.equ2pers.GetPerspective(image, YAW, PITCH)
        kp = self.orb.detect(image, None)

        # train_image = self.process(image)
        # scores = self.local_feature_extractor(train_image)['scores']
        # num_valid=np.sum(np.where(scores>0.005))
        valid=True
        if len(kp) <= 100:
            valid=False
        descriptors = {}
        for name, extractor in self.extractors.items():
            descriptors.update({name: extractor.feature(image)[0]})
        return descriptors,valid

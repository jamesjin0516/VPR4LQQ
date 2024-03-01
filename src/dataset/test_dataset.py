import numpy as np
from os.path import join,exists,basename
from os import listdir
from typing import Optional, Sized
from dataset.GT_pose import GT_pose

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Sampler, ConcatDataset

from sklearn.neighbors import NearestNeighbors
import json
from PIL import Image
from tqdm import tqdm
import h5py
import random


def read_coordinates(image_names):    # -> shape = (len(image_names), 2])
    all_coordinates = []
    for i in image_names:
        coor = read_coordinate(i)
        all_coordinates.append(coor)
    return np.array(all_coordinates)
    
def read_coordinate(image_name):
    parts = image_name.split("@")
    x = parts[1]
    y = parts[2]
    return [x, y]

class TestDataset(Dataset):

    def __init__(self, image_folder, resolution, test_data_config, train_loss_config):
        """
        - image_folder: path to query images folder (one level above individual resolutions folder)
        - resolution: specifies the resolution folder to choose for query images
        """
        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
        ])
        self.image_folder = image_folder
        self.resolution = resolution 
        self.nPosSample,self.nNegSample = train_loss_config['nPosSample'],train_loss_config['nNegSample'] # 1, 5
        self.neighbor_file = h5py.File(join(image_folder, 'neighbors.h5'), 'r')
        self.__prepare_data(image_folder)


    def __getitem__(self, index):
        high_image_set,low_image_set=self.data['data'][index]
        paths = self.data['path'][index]

        images_high,images_low,locations=[],[],[]
        image_high,positives_high,negtives_high=high_image_set
        image_low,positives_low,negtives_low=low_image_set
        for im in [image_high] + positives_high + negtives_high:
            images_high.append(im)
        images_high = torch.stack(images_high)
        for im in [image_low] + positives_low + negtives_low:
            images_low.append(im)
        images_low = torch.stack(images_low)

        for imp in paths:
            coor = read_coordinate(basename(imp))
            locations.append(coor)

        locations = np.array(locations)
        locations = torch.tensor(locations)
        assert locations.shape[0] == 1 + self.nPosSample + self.nNegSample

        return [images_high, images_low, locations]
        # tensors, with shape [(7, #, #, #), (7, #, #, #), (7, 2)]

    def __len__(self):
        return len(self.data['data'])

    def __prepare_data(self,image_folder):
        self.data={}
        self.data['data']=[]
        self.data['path']=[]

        image_list=listdir(image_folder)
        image_list.remove('global_descriptor.h5')
        image_list.remove('neighbors.h5')
        self.image_coordinates = read_coordinates(image_list)
        for image in image_list:
            if image in self.neighbor_file:
                image_high_path=join(image_folder, 'raw', image)
                image_high = self.input_transform(Image.open(image_high_path))
                image_low_path=join(image_folder, self.resolution, image)
                image_low = self.input_transform(Image.open(image_low_path))
                
                positives_high,negatives_high=[],[]
                positives_low,negatives_low=[],[]
                paths=[]
                paths.append(image_high_path)

                positives_pool=self.neighbor_file[name]['positives'][:] #(20,)
                negatives_pool=self.neighbor_file[name]['negtives'][:] #(100,)

                ind=0
                while len(positives_high) < self.nPosSample:
                    name=positives_pool[ind].decode('utf-8')
                    positive_high_path=join(image_folder, 'raw', name)
                    positive_low_path=join(image_folder, self.resolution, name)
                    if exists(positive_high_path):
                        positive_high = self.input_transform(Image.open(positive_high_path))
                        positive_low = self.input_transform(Image.open(positive_low_path))
                        positives_high.append(positive_high)
                        positives_low.append(positive_low)
                        paths.append(positive_high_path)
                        
                    ind += 1
                    if ind==len(positives_pool)-1:
                        break
                
                ind=0
                while len(negatives_high) < self.nNegSample:
                    name=negatives_pool[ind].decode('utf-8')
                    negative_high_path=join(image_folder, 'raw', name)
                    negative_low_path=join(image_folder, self.resolution, name)
                    if exists(negative_high_path):
                        negative_high = self.input_transform(Image.open(negative_high_path))
                        negative_low = self.input_transform(Image.open(negative_low_path))
                        negatives_high.append(negative_high)
                        negatives_low.append(negative_low)
                        paths.append(negative_high_path)
                        
                    ind += 1
                    if ind==len(negatives_pool)-1:
                        break
                if len(positives_high)==self.nPosSample and len(negatives_high)==self.nNegSample:
                    self.data['data'].append([[image_high,positives_high,negatives_high],[image_low,positives_low,negatives_low]])
                    self.data['path'].append(paths)

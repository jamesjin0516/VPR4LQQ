from os.path import join,exists,basename
from os import listdir
from typing import Optional, Sized
import numpy as np
import json
from torch.utils.data import ConcatDataset
from dataset.GT_pose import GT_pose
from torch.utils.data import Dataset,Sampler
from sklearn.neighbors import NearestNeighbors
import json
from PIL import Image
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import h5py
import random


class Pitts250k_dataset(Dataset):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    def __init__(self,image_folder,resolution,neighbor_file,gt,image_id,**config):
        self.gt=gt # (1000, 2)
        self.id=image_id # 1000
        self.neighbor_file=neighbor_file
        self.nPosSample,self.nNegSample=config['nPosSample'],config['nNegSample']
        self.high_resolution='raw'
        self.resolution=resolution
        self.__prepare_data(image_folder)
        
    def __getitem__(self, index):
        high_image_set,low_image_set=self.data[index]
        image_high_path,positives_high,negtives_high=high_image_set
        image_low_path,positives_low,negtives_low=low_image_set
        images_high,images_low,images_low_path,locations=[],[],[],[]
        for im in [image_high_path]+positives_high+negtives_high:
            images_high.append(self.input_transform()(Image.open(im)))
        images_high=torch.stack(images_high)
        for im in [image_low_path]+positives_low+negtives_low:
            images_low_path.append(im)
            images_low.append(self.input_transform()(Image.open(im)))
            locations.append(self.gt[self.id.index(int(basename(im).replace('.jpg','')))])
        images_low=torch.stack(images_low)
        locations=np.array(locations)
        locations=torch.tensor(locations)
        return [images_high,images_low,images_low_path,locations]
    
    def input_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __prepare_data(self,image_folder):
        self.data=[]
        pitch_list=listdir(image_folder)
        pitch_list.remove('global_descriptor.h5')
        pitch_list.remove('neighbors.h5')
        for pitch in sorted(pitch_list):
            pitch_folder=join(image_folder,pitch)
            yaw_list=listdir(pitch_folder)
            for yaw in sorted(yaw_list):
                images_root=join(pitch_folder,yaw)
                images_low_path=join(images_root,self.resolution)
                images_high_path=join(images_root,'raw')

                images_low=listdir(images_low_path)
                for image in images_low:
                    name=f'{pitch}+{yaw}+{image}'
                    if name in self.neighbor_file:
                        image_high_path=join(images_high_path,image)
                        image_low_path=join(images_low_path,image)
                        positives_high,negtives_high=[],[]
                        positives_low,negtives_low=[],[]
                        ind=0
                        positives_pool=self.neighbor_file[name]['positives'][:]
                        negtives_pool=self.neighbor_file[name]['negtives'][:]
                        while len(positives_high)<self.nPosSample:
                            pitch_,yaw_,name_=positives_pool[ind].decode('utf-8').split('+')
                            positive_high=join(image_folder,pitch_,yaw_,self.high_resolution,name_)
                            positive_low=join(image_folder,pitch_,yaw_,self.resolution,name_)
                            if exists(positive_high) and exists(positive_low):
                                positives_high.append(positive_high)
                                positives_low.append(positive_low)
                            ind+=1
                            if ind==len(positives_pool)-1:
                                break
                        ind=0
                        while len(negtives_high)<self.nNegSample:
                            pitch_,yaw_,name_=negtives_pool[ind].decode('utf-8').split('+')
                            negtive_high=join(image_folder,pitch_,yaw_,self.high_resolution,name_)
                            negtive_low=join(image_folder,pitch_,yaw_,self.resolution,name_)
                            if exists(negtive_high) and exists(negtive_low):
                                negtives_high.append(negtive_high)
                                negtives_low.append(negtive_low)
                            ind+=1
                            if ind==len(negtives_pool)-1:
                                break
                        if len(positives_high)==self.nPosSample and len(negtives_high)==self.nNegSample:
                            self.data.append([[image_high_path,positives_high,negtives_high],[image_low_path,positives_low,negtives_low]])

def load_pitts250k_data(data,config):
    image_folder=data['image_folder']
    gt=data['utm'] # (1000, 2) eg. [585001.41335051, 4477058.99275442]
    image_id=data['id'] # 1000

    neighbor_file = h5py.File(join(image_folder,'neighbors.h5'), 'r')

    pitch_list=listdir(image_folder)
    pitch_list.remove('global_descriptor.h5')
    pitch_list.remove('neighbors.h5')
    
    for pitch in sorted(pitch_list):
        pitch_folder=join(image_folder,pitch)
        yaw_list=listdir(pitch_folder)
        for yaw in yaw_list:
            images_root=join(pitch_folder,yaw)
            resolution_list=sorted(listdir(images_root))
            break
        break

    dataset=[]
    dataconfig=config['data']
    if dataconfig['resolution']==-1:
        for resolution in tqdm(resolution_list,desc=f'loading data from {image_folder}',total=len(resolution_list)):
            if resolution!='raw':
                dataset.append(Pitts250k_dataset(image_folder,resolution,neighbor_file,gt,image_id,**config['triplet_loss']))
    else:
        dataset.append(Pitts250k_dataset(image_folder,dataconfig['resolution'],neighbor_file,gt,image_id,**config['triplet_loss']))

    return dataset

def load_data(image_folder,**config):
    pitch_list=listdir(image_folder)
    pitch_list.remove('global_descriptor.h5')
    pitch_list.remove('neighbors.h5')
    
    for pitch in sorted(pitch_list):
        pitch_folder=join(image_folder,pitch)
        yaw_list=listdir(pitch_folder)
        for yaw in yaw_list:
            images_root=join(pitch_folder,yaw,'images')
            resolution_list=sorted(listdir(images_root))
            qp_list=sorted(listdir(join(images_root,resolution_list[0])))
            break
        break

    dataset=[]
    dataconfig=config['data']
    if dataconfig['resolution']==-1:
        for i,resolution in tqdm(enumerate(resolution_list),desc=f'loading data from {image_folder}',total=len(resolution_list)):
            if dataconfig['qp']==-1:
                if i==len(resolution_list)-1:
                    qp_list.remove('raw')
                for qp in qp_list:
                    dataset.append(UNav_dataset(image_folder,resolution,qp,pitch_list,yaw_list,**config['triplet_loss']))
            else:
                dataset.append(UNav_dataset(image_folder,resolution,dataconfig['qp'],pitch_list,yaw_list,**config['triplet_loss']))
    else:
        if dataconfig['qp']==-1:
            for qp in qp_list:
                dataset.append(UNav_dataset(image_folder,dataconfig['resolution'],qp,pitch_list,yaw_list,**config['triplet_loss']))
        else:
            dataset.append(UNav_dataset(image_folder,dataconfig['resolution'],dataconfig['qp'],pitch_list,yaw_list,**config['triplet_loss']))

    return dataset

def load_test_data(image_folder,**config):
    pitch_list=listdir(image_folder)
    pitch_list.remove('global_descriptor.h5')
    pitch_list.remove('neighbors.h5')
    for pitch in sorted(pitch_list):
        pitch_folder=join(image_folder,pitch)
        yaw_list=listdir(pitch_folder)
        for yaw in yaw_list:
            images_root=join(pitch_folder,yaw,'images')
            resolution_list=sorted(listdir(images_root))
            qp_list=sorted(listdir(join(images_root,resolution_list[0])))
            break
        break
    dataset={}
    for i,resolution in tqdm(enumerate(resolution_list),desc=f'loading data from {image_folder}',total=len(resolution_list)):
        if resolution not in dataset:
            dataset[resolution]={}
        for qp in qp_list:
            dataset[resolution][qp]=UNav_dataset(image_folder,resolution,qp,pitch_list,yaw_list,**config['triplet_loss'])
    return dataset

def data_link(root,name,scales,data_list):
    data={}
    for d in data_list:
        key=d.split('_')[0]
        if key in scales:
            data[d]={'image_folder':join(root,'logs',name,d),'utm_file':join(root,'data',name,'utm',d+'.json'),'scale':scales[key]}
    return data

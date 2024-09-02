from os.path import join, exists, basename, isdir
from os import listdir
import numpy as np
import json
from torch.utils.data import Dataset,Sampler
import json
from PIL import Image
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import h5py
import random

def chunk(indices, size):
    return torch.split(torch.tensor(indices),size)

class BatchSampler(Sampler):
    def __init__(self, data_indices, batch_size):
        self.data_indices=data_indices
        self.batch_size=batch_size
    
    def __iter__(self):
        data_batches=[]
        for d in self.data_indices:
            random.shuffle(d)
            data_batches.append(chunk(d,self.batch_size))
        d=data_batches[0]
        for dd in data_batches[1:]:
            d+=dd
        all_batches=list(d)
        all_batches=[batch.tolist() for batch in all_batches]
        random.shuffle(all_batches)
        return iter(all_batches)
    
    def __len__(self):
        return sum([len(d) for d in self.data_indices])/self.batch_size
    
class UNav_dataset(Dataset):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    def __init__(self,image_folder,resolution,qp,pitch_list,yaw_list,**config):
        self.nPosSample,self.nNegSample=config['nPosSample'],config['nNegSample']
        num=0
        for pitch in pitch_list:
            for yaw in yaw_list:
                neighbor_folder=join(image_folder,pitch,yaw,'neighbor',resolution,str(qp))
                num+=len(listdir(neighbor_folder))
        self.data=np.empty(num,dtype='U100')
        index=0
        for pitch in pitch_list:
            for yaw in yaw_list:
                neighbor_folder=join(image_folder,pitch,yaw,'neighbor',resolution,str(qp))
                datas=listdir(neighbor_folder)
                for data in datas:
                    self.data[index]=join(neighbor_folder,data)
                    index+=1
        
    def __getitem__(self, index):
        with open(self.data[index],'r') as f:
            data=json.load(f)
        locs=data['locs']
        data=data['paths']
        images_high_dict=data['high']
        image_high_path=images_high_dict['path']
        positives_high=images_high_dict['positives']
        negtives_high=images_high_dict['negtives']

        images_low_dict=data['low']
        image_low_path=images_low_dict['path']
        positives_low=images_low_dict['positives']
        negtives_low=images_low_dict['negtives']

        images_low_path,images_high,images_low=[],[],[]
        for im in [image_high_path]+positives_high+negtives_high:
            images_high.append(self.input_transform()(Image.open(im)))
        images_high=torch.stack(images_high)
        for im in [image_low_path]+positives_low+negtives_low:
            images_low_path.append(im)
            images_low.append(self.input_transform()(Image.open(im)))
        images_low=torch.stack(images_low)
        locations=torch.tensor(locs)
        return [images_high,images_low,images_low_path,locations]
    
    def input_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    reorganized = [dict() for _ in range(4)]    # images_high, images_low, images_low_path, locations for each model
    for batch_item in batch:
        for model, image_data in batch_item.items():
            # Distribute each type of image_data into the reorganized batch, where each type is still grouped into models
            for i in range(len(image_data)):
                if model in reorganized[i]:
                    reorganized[i][model].append(image_data[i])
                else:
                    reorganized[i][model] = [image_data[i]]
    for model, low_paths in reorganized[2].items():
        reorganized[2][model] = [tuple(low_paths[batch_ind][path_ind] for batch_ind in range(len(low_paths))) for path_ind in range(len(low_paths[0]))]
    for data_ind in range(len(reorganized)):
        if data_ind == 2: continue
        for model in reorganized[data_ind]:
            reorganized[data_ind][model] = torch.stack(reorganized[data_ind][model])
    return reorganized


class Pitts250k_dataset(Dataset):
    
    def __init__(self, image_folder, resolution, models, gt, image_id, **config):
        self.gt=gt # (1000, 2)
        self.id=image_id # 1000
        self.nPosSample,self.nNegSample=config['nPosSample'],config['nNegSample'] # 1, 5
        self.high_resolution='raw'
        self.resolution = resolution
        self.input_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        neighbor_files = {model: h5py.File(join(image_folder, f"neighbors_{model}.h5"), "r") for model in models}
        self.__prepare_data(image_folder, neighbor_files)
        for file in neighbor_files.values(): file.close()
        
    def __getitem__(self, index):
        image_data = {}
        for model, (high_image_set, low_image_set) in self.data[index].items():
            # concat image, postive image, and negative images
            image_high_path, positives_high, negtives_high = high_image_set
            image_low_path, positives_low, negtives_low = low_image_set
            images_high, images_low, images_low_path, locations = [], [], [], []
            for im in [image_high_path] + positives_high + negtives_high:
                images_high.append(self.input_transform(Image.open(im)))
            images_high=torch.stack(images_high)
            for im in [image_low_path] + positives_low + negtives_low:
                images_low_path.append(im)
                images_low.append(self.input_transform(Image.open(im)))
                locations.append(self.gt[self.id.index(int(basename(im).rstrip(".jpg")))]) # gt corresponding to the image path
            images_low=torch.stack(images_low)
            locations = torch.tensor(np.array(locations))
            image_data[model] = [images_high, images_low, images_low_path, locations]
            # images_high: (7, 3, 480, 640)
            # images_low: (7, 3, 180, 240)
            # images_low_path: 7 (list)
            # locations: (7, 2)
        return image_data
    
    def __len__(self):
        return len(self.data)
    
    def __prepare_data(self, image_folder, neighbor_files):
        self.data, discarded_imgs = [], []
        dir_contents, pitch_list = listdir(image_folder), []
        for dir_entry in dir_contents:
            if isdir(join(image_folder, dir_entry)):
                pitch_list.append(dir_entry)
        for pitch in sorted(pitch_list):
            pitch_folder=join(image_folder,pitch)
            yaw_list=listdir(pitch_folder)
            for yaw in sorted(yaw_list):
                images_root=join(pitch_folder,yaw)
                images_low_path=join(images_root,self.resolution) # '/scratch/lg3490/VPR4LQQ/logs/pitts250k/query/000/000/180p'
                images_high_path=join(images_root,'raw') # '/scratch/lg3490/VPR4LQQ/logs/pitts250k/query/000/000/raw'

                images_low=listdir(images_low_path)
                for image in images_low:
                    name, image_data = f"{pitch}+{yaw}+{image}", {}
                    for model, neighbor_file in neighbor_files.items():
                        image_high_path=join(images_high_path,image) # '/scratch/lg3490/VPR4LQQ/logs/pitts250k/query/000/000/raw/006420.jpg'
                        image_low_path=join(images_low_path,image)
                        positives_high,negtives_high=[],[]
                        positives_low,negtives_low=[],[]
                        ind=0
                        # key: how to get the neighbor thing in the new dataset
                        positives_pool = neighbor_file[name]['positives'][:] #(20,)
                        negtives_pool = neighbor_file[name]['negtives'][:] #(100,)
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
                            image_data[model] = [[image_high_path, positives_high, negtives_high], [image_low_path, positives_low, negtives_low]]
                        # image_high_path: '/scratch/lg3490/VPR4LQQ/logs/pitts250k/query/000/000/raw/006420.jpg'
                        # positives_high: ['/scratch/lg3490/VPR4LQQ/logs/pitts250k/query/000/000/raw/006419.jpg']
                        # negtives_high: ['/scratch/lg3490/VPR4LQQ/logs/pitts250k/query/000/210/raw/006352.jpg', '/scratch/lg3490/VPR4LQQ/logs/pitts250k/query/030/240/raw/003258.jpg', '/scratch/lg3490/VPR4LQQ/logs/pitts250k/query/030/300/raw/008770.jpg', '/scratch/lg3490/VPR4LQQ/logs/pitts250k/query/000/180/raw/006603.jpg', '/scratch/lg3490/VPR4LQQ/logs/pitts250k/query/000/060/raw/008609.jpg']
                        # same pattern for low
                    if len(image_data) == len(neighbor_files):
                        self.data.append(image_data)
                    else:
                        discarded_imgs.append(name)
        if len(discarded_imgs) > 0:
            print(f"Pitts250k dataset ({basename(image_folder)}) discarded {len(discarded_imgs)} for insufficient"
                  f"neighbors from at least one extractor.\n{discarded_imgs}")


def collate_fn_with_high_descriptors(batch):
    reorganized = [dict() for _ in range(5)]    # images_high, high_descriptors, images_low, images_low_path, locations for each model
    for batch_item in batch:
        for model, image_data in batch_item.items():
            # Distribute each type of image_data into the reorganized batch, where each type is still grouped into models
            for i in range(len(image_data)):
                if model in reorganized[i]:
                    reorganized[i][model].append(image_data[i])
                else:
                    reorganized[i][model] = [image_data[i]]
    for model, low_paths in reorganized[3].items():
        reorganized[3][model] = [tuple(low_paths[batch_ind][path_ind] for batch_ind in range(len(low_paths))) for path_ind in range(len(low_paths[0]))]
    for data_ind in range(len(reorganized)):
        if data_ind == 3: continue
        for model in reorganized[data_ind]:
            reorganized[data_ind][model] = torch.stack(reorganized[data_ind][model])
    return reorganized


class GSVCitiesDataset(Dataset):
    high_resolution = "480p"
    def __init__(self, images_path, resolution, cities, models, groundtruth, **config):
        self.groundtruth = groundtruth
        self.input_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.__prepare_data(images_path, resolution, cities, models, config["nPosSample"], config["nNegSample"])
        self.descriptor_files = {city: {model: h5py.File(join(images_path, city, f"global_descriptor_{model}.h5"), "r") for model in models} for city in cities}
        
    def __getitem__(self, index):
        image_data = {}
        for model, (high_image_set, low_image_set) in self.data[index].items():
            # concat image, postive image, and negative images
            image_high_path, positives_high, negtives_high = high_image_set
            image_low_path, positives_low, negtives_low = low_image_set
            city = basename(image_high_path).split("_")[0]
            images_high, high_descriptors, images_low, images_low_path, locations = [], [], [], [], []
            for im in [image_high_path] + positives_high + negtives_high:
                images_high.append(self.input_transform(Image.open(im)))
                # high_descriptors.append(torch.tensor(self.descriptor_files[city][model]["train"][basename(im)][:]))
            images_high = torch.stack(images_high)
            for im in [image_low_path] + positives_low + negtives_low:
                images_low_path.append(im)
                images_low.append(self.input_transform(Image.open(im)))
                locations.append(self.groundtruth[basename(im)]) # gt corresponding to the image path
            images_low = torch.stack(images_low)
            # high_descriptors = torch.stack(high_descriptors)
            locations = torch.tensor(np.array(locations))
            image_data[model] = [images_high, images_low, images_low_path, locations]
        return image_data
    
    def __len__(self):
        return len(self.data)
    
    def close_descriptor_files(self):
        for city_files in self.descriptor_files.values():
            for model_file in city_files.values():
                model_file.close()
    
    def __prepare_data(self, image_folder, resolution, cities, models, nPosSample, nNegSample):
        self.data, discarded_imgs = [], []
        for city in cities:
            city_folder = join(image_folder, city)
            neighbor_files = {model: h5py.File(join(city_folder, f"neighbors_{model}.h5"), "r") for model in models}
            images_low_path, images_high_path = join(city_folder, resolution), join(city_folder, self.high_resolution)
            # For every image, load postives & negatives found by each model 
            for img_name in listdir(images_low_path):
                image_data = {}
                for model, neighbor_file in neighbor_files.items():
                    image_high_path, image_low_path = join(images_high_path, img_name), join(images_low_path, img_name)
                    positives_high, negtives_high = [], []
                    positives_low, negtives_low = [], []
                    ind = 0
                    positives_pool = neighbor_file[img_name]["positives"][:]
                    negtives_pool = neighbor_file[img_name]["negtives"][:]
                    while len(positives_high) < nPosSample and ind < len(positives_pool):
                        positive_name = positives_pool[ind].decode("utf-8")
                        positive_high = join(city_folder, self.high_resolution, positive_name)
                        positive_low = join(city_folder, resolution, positive_name)
                        positives_high.append(positive_high)
                        positives_low.append(positive_low)
                        ind += 1
                    ind = 0
                    while len(negtives_high) < nNegSample and ind < len(negtives_pool):
                        negative_name = negtives_pool[ind].decode("utf-8")
                        negtive_high = join(city_folder, self.high_resolution, negative_name)
                        negtive_low = join(city_folder, resolution, negative_name)
                        negtives_high.append(negtive_high)
                        negtives_low.append(negtive_low)
                        ind += 1
                    if len(positives_high) == nPosSample and len(negtives_high) == nNegSample:
                        image_data[model] = [[image_high_path, positives_high, negtives_high], [image_low_path, positives_low, negtives_low]]
                if len(image_data) == len(neighbor_files):
                    self.data.append(image_data)
                else:
                    discarded_imgs.append(img_name)
            for file in neighbor_files.values(): file.close()
        if len(discarded_imgs) > 0:
            print(f"GSV-Cities dataset discarded {len(discarded_imgs)} for insufficient"
                  f"neighbors from at least one extractor.\n{discarded_imgs}")


def load_pitts250k_data(data, config, models):
    image_folder=data['image_folder']
    gt=data['utm'] # (1000, 2) eg. [585001.41335051, 4477058.99275442]
    image_id=data['id'] # 1000

    dir_contents, pitch_list = listdir(image_folder), []
    for dir_entry in dir_contents:
        if isdir(join(image_folder, dir_entry)):
            pitch_list.append(dir_entry)
    
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
                dataset.append(Pitts250k_dataset(image_folder, resolution, models, gt, image_id, **config["triplet_loss"]))
    else:
        dataset.append(Pitts250k_dataset(image_folder, dataconfig["resolution"], models, gt, image_id, **config["triplet_loss"]))

    return dataset


def load_gsv_cities_data(dataset_info, train_conf, cities, models):
    groundtruth, images_path = dataset_info["groundtruth"], dataset_info["images_path"]
    resolutions = []
    for dir_entry in listdir(join(images_path, cities[0])):
        if isdir(join(images_path, cities[0], dir_entry)):
            resolutions.append(dir_entry)
    resolutions = sorted(resolutions)

    datasets = []
    if train_conf["data"]["resolution"] == -1:
        for resolution in resolutions:
            if resolution != "raw":
                datasets.append(GSVCitiesDataset(images_path, resolution, cities, models, groundtruth, **train_conf["triplet_loss"]))
    else:
        datasets.append(GSVCitiesDataset(images_path, train_conf["data"]["resolution"], cities, models, groundtruth, **train_conf["triplet_loss"]))
    return datasets


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

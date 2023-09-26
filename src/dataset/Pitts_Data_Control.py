import os
from os.path import basename
import glob
import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


class Pitts_dataset(Dataset):
    IMAGE_EXT = ".png"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, data_path, triplet_loss_config, prepared_dataset, global_extractor, **configs):
        self.global_descriptor_dim = configs['train']['num_cluster'] * configs['train']['cluster']['dimension']
        self.posDistThr = triplet_loss_config['posDistThr']
        self.nonTrivPosDistSqThr = triplet_loss_config['nonTrivPosDistSqThr']
        self.nNegSample = triplet_loss_config['nNegSample']
        self.prepared_dataset = os.path.join(prepared_dataset, "prepared_dataset.json")
        self.global_extractor = global_extractor
        self.images_name = []
        self.data = self.__prepare_data(data_path)
        
        

    def __len__(self):
        # TODO: Implement this method based on the dataset's size
        return len(self.data)

    def __getitem__(self, index):
        # TODO: Implement this method to retrieve a specific item from the dataset
        return self.data[index]

    def __get_image_paths_from_dir(self, directory):
        image_files = glob.glob(os.path.join(directory, '*.jpg'))
        print(directory,"images have been found")
        return image_files

    def __neighbor(self, image_folder):
        locations = []
        for order in ["000"]:
            path = os.path.join("/mnt/data/Resolution_Agnostic_VPR/logs/Pitts250k", order + ".json")
            location = self.__get_location(path)
            print(order, "location get")
            locations.extend(location)
        nontrivial_positives = self.get_nontrivial_positives(locations)
        print("nontrivial_positives get, ",nontrivial_positives[0])
        potential_negatives = self.get_potential_negatives(locations, nontrivial_positives)
        print("potential_negatives get, ",potential_negatives[0])
        print("neighbor finished")
        return nontrivial_positives, potential_negatives

    def get_nontrivial_positives(self, locations):
        knn = NearestNeighbors(n_jobs=1).fit(locations)
        nontrivial_positives = list(knn.radius_neighbors(locations, radius=self.nonTrivPosDistSqThr, return_distance=False))
        
        for i, posi in enumerate(nontrivial_positives):
            nontrivial_positives[i] = np.sort(posi)
        print(81)
        return nontrivial_positives

    def get_potential_negatives(self, locations, nontrivial_positives):
        knn = NearestNeighbors(n_jobs=1).fit(locations)
        potential_positives = knn.radius_neighbors(locations, radius=self.posDistThr, return_distance=False)
        
        potential_negatives = []
        for pos in potential_positives:
            potential_negatives.append(np.setdiff1d(np.arange(len(locations)), pos, assume_unique=True))
        return potential_negatives

    def __prepare_data(self, data_path):
        data = []
        for key, value in data_path.items():
            # Simplified variable naming for clarity
            '''
            data_path:
            Tandon4_0 {'image_folder': '/mnt/data/Resolution_Agnostic_VPR/logs/unav/Tandon4_0', 
            'utm_file': '/mnt/data/Resolution_Agnostic_VPR/data/unav/utm/Tandon4_0.json', 
            'scale': 0.01209306372}
            '''
            image_folder = "/mnt/data/Resolution_Agnostic_VPR/logs/Pitts250k"
            
            self.positives, self.negatives = self.__neighbor(image_folder)
        
            for order in sorted(os.listdir(image_folder)):
                print(f"processing {order}")
                for image in ["000"]:
                    images_root = os.path.join(image_folder, image)
                images_high_path = self.__get_image_paths_from_dir(os.path.join(images_root))
                pair = self.__get_pairs(images_high_path)
                data+=pair
                print(113)
            
        return data

    def __get_pairs(self, images_high_path):
        for i in range(len(images_high_path)):
            high_reso_img = cv2.imread(images_high_path[i])
            
            scale_factor = np.random.uniform(0.1, 1)
            random_height = int(high_reso_img.shape[0] * scale_factor)
            random_width = int(high_reso_img.shape[1] * scale_factor)
            low_reso_img = cv2.resize(high_reso_img,(random_height, random_width))
            pairs = [high_reso_img,low_reso_img]
        return pairs


    def __get_location(self, json_path):
        with open(json_path, 'r') as file:
            parsed_data = json.load(file)
            array_2d = [[item['x'], item['y']] for key, item in parsed_data.items()]
            location = np.array(array_2d)
        return location
    
    def extract_descriptors(image_folder,global_extractor):
        hfile_path=join(image_folder,'global_descriptor.h5')
        if not exists(hfile_path):
            hfile = h5py.File(hfile_path, 'a')
            grp = hfile.create_group(basename(image_folder))
            index=0
            pitch_list=listdir(image_folder)
            for pitch in sorted(pitch_list):
                pitch_folder=join(image_folder,pitch)
                yaw_list=listdir(pitch_folder)
                for ind,yaw in enumerate(sorted(yaw_list)):
                    images_root=join(pitch_folder,yaw,'images')
                    resolution_list=sorted(listdir(images_root))
                    images_high_path=join(images_root,resolution_list[-1],'raw')
                    image_high_names = set(sorted(listdir(images_high_path)))
                    for i,im in tqdm(enumerate(image_high_names),desc=f'{str(ind).zfill(2)}/{len(yaw_list)}',total=len(image_high_names)):
                        if i%30==0 or i==len(image_high_names)-1:
                            if i>0:
                                image_=torch.cat(images_list)
                                feature=global_extractor.encoder(image_)
                                vector=global_extractor.pool(feature).detach().cpu()
                                for name, descriptor in zip(images_name,vector):

                                    grp.create_dataset(name, data=descriptor)

                                index+=vector.size(0)
                                del image_,feature,vector
                                torch.cuda.empty_cache()
                            images_list=[]
                            images_name=[]
                        image=cv2.imread(join(images_high_path,im))
                        image=torch.tensor(image).permute(2,0,1).unsqueeze(0).float().to(self.device)
                        images_list.append(image)
                        images_name.append(f'{pitch}+{yaw}+{im}')
            hfile.close()



def data_link(root, name, scales, data_list):
    datalist = ["000"]
    return {d: {'image_folder': os.path.join(root, 'logs', "Pitts250k", d),
                'utm_file': os.path.join(root, 'logs', "Pitts250k", f"{d}.json"),
                'scale': scales.get(d.split('_')[0], 1)}
            for d in data_list}


# TODO: Add proper invocation, testing and main entry point if required.

import sys
from os.path import join,exists,basename
from os import listdir
sys.path.append(join(sys.path[0],'..'))
import numpy as np
from sklearn.neighbors import NearestNeighbors
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import h5py
from third_party.pytorch_NetVlad.Feature_Extractor import NetVladFeatureExtractor
import argparse
import yaml
import json
import random

def get_gt(aligner_path):
    with open(aligner_path, "r") as f:
        keyframes = json.load(f)['keyframes']
    key= {}
    for id, point in keyframes.items():
        t_mp=point['trans']
        rot=point['rot']
        key[str(int(id)-1).zfill(5)] = [t_mp[0], t_mp[1], rot]
    return key

def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
    ])

def extract_descriptors(image_folder,global_extractor):
    hfile_path=join(image_folder,'global_descriptor.h5')
    if not exists(hfile_path):
        pitch_list=listdir(image_folder)
        hfile = h5py.File(hfile_path, 'a')
        grp = hfile.create_group(basename(image_folder))
        index=0
        for pitch in sorted(pitch_list):
            pitch_folder=join(image_folder,pitch)
            yaw_list=listdir(pitch_folder)
            for ind,yaw in enumerate(sorted(yaw_list)):
                images_root=join(pitch_folder,yaw,'images')
                resolution_list=sorted(listdir(images_root))
                images_high_path=join(images_root,resolution_list[-1],'raw')
                image_high_names = set(sorted(listdir(images_high_path)))
                for i,im in tqdm(enumerate(image_high_names),desc=f'{str(ind).zfill(2)}/{len(yaw_list)}',total=len(image_high_names)):
                    if i%20==0 or i==len(image_high_names)-1:
                        if i>0:
                            image_=torch.stack(images_list)
                            feature=global_extractor.encoder(image_)
                            vector=global_extractor.pool(feature).detach().cpu()
                            for name, descriptor in zip(images_name,vector):
                                grp.create_dataset(name, data=descriptor)
                            index+=vector.size(0)
                            del image_,feature,vector
                            torch.cuda.empty_cache()
                        images_list=[]
                        images_name=[]
                    image=input_transform()(Image.open(join(images_high_path,im)))
                    images_list.append(image.to(device))
                    images_name.append(f'{pitch}+{yaw}+{im}')
        hfile.close()

def find_neighbors(image_folder,gt,scale,global_descriptor_dim,posDistThr,nonTrivPosDistSqThr,nPosSample):

    pitch_list=listdir(image_folder)
    pitch_list.remove('global_descriptor.h5')
    for pitch in sorted(pitch_list):
        pitch_folder=join(image_folder,pitch)
        yaw_list=listdir(pitch_folder)

    hfile_path=join(image_folder,'global_descriptor.h5')
    hfile = h5py.File(hfile_path, 'r')
    
    names=[]
    descriptors=np.empty((len(hfile[basename(image_folder)]),global_descriptor_dim))
    i=0
    for k,v in hfile[basename(image_folder)].items():
        names.append(k)
        descriptors[i,:]=v.__array__()
        i+=1
    descriptors=torch.from_numpy(descriptors)
    sim = torch.einsum('id,jd->ij', descriptors, descriptors).float()
    ind = np.diag_indices(sim.shape[0])
    sim[ind[0], ind[1]] = torch.zeros(sim.shape[0],dtype=torch.float)
    topk = torch.topk(sim, descriptors.size(0), dim=1).indices.numpy()

    locations=[]
    name_ori=set([name.replace('.png','').split('+')[-1] for name in names])
    name_phisical=[]
    for k,v in gt.items():
        if k in name_ori:
            name_phisical.append(k)
            locations.append(v)

    locations=np.array(locations)*scale
    knn = NearestNeighbors(n_jobs=1)
    knn.fit(locations)
    nontrivial_positives = list(knn.radius_neighbors(locations,
            radius=nonTrivPosDistSqThr, 
            return_distance=False))

    potential_positives = knn.radius_neighbors(locations,
            radius=posDistThr, 
            return_distance=False)
    potential_negatives = []
    for pos in potential_positives:
        potential_negatives.append(np.setdiff1d(np.arange(len(locations)),
            pos, assume_unique=True))

    hfile_neighbor_path=join(image_folder,'neighbors.h5')
    hfile = h5py.File(hfile_neighbor_path, 'a')

    for i,name in enumerate(names):
        key=name.replace('.png','').split('+')[-1]
        physical_closed=set([name_phisical[ind] for ind in nontrivial_positives[name_phisical.index(key)]])

        negtives_=[name_phisical[ind] for ind in potential_negatives[name_phisical.index(key)]]
        negtives=[]
        for pitch in pitch_list:
            for yaw in yaw_list:
                for neg in negtives_:
                    negtives.append(f'{pitch}+{yaw}+{neg}.png')

        negtives=random.sample(negtives,100)

        feature_closed=topk[i]
        positives=[]
        ind=0

        while len(positives)<nPosSample*20:
            key=names[feature_closed[ind]].replace('.png','').split('+')[-1]
            if key in physical_closed:
                positives.append(names[feature_closed[ind]])
            ind+=1
            if ind>=len(feature_closed):
                break

        if len(positives)>0:
            grp = hfile.create_group(name)
            grp.create_dataset('positives', data=positives)
            grp.create_dataset('negtives',data=negtives)

    hfile.close()
    return nontrivial_positives,potential_negatives

def main(configs):
    root=configs['root']
    content=configs['vpr']['global_extractor']['netvlad']
    teacher_model=NetVladFeatureExtractor(join(configs['root'], content['ckpt_path']), arch=content['arch'],
        num_clusters=content['num_clusters'],
        pooling=content['pooling'], vladv2=content['vladv2'], nocuda=content['nocuda'])
    teacher_model.model.to(device).eval()
    image_folders=sorted(listdir(join(root,'logs','unav')))

    with open(join(root,'configs','measurement_scale.yaml'), 'r') as f:
        scales = yaml.safe_load(f)

    global_descriptor_dim=configs['train']['num_cluster']*configs['train']['cluster']['dimension']
    posDistThr=configs['train']['triplet_loss']['posDistThr']
    nonTrivPosDistSqThr=configs['train']['triplet_loss']['nonTrivPosDistSqThr']
    nPosSample=configs['train']['triplet_loss']['nPosSample']
    image_folders=['Lighthouse6_0','Lighthouse6_1','Lighthouse6_2','Lighthouse6_3','Lighthouse6_4','Lighthouse6_5','Lighthouse6_6','Lighthouse6_7','Tandon4_0']
    for image_folder in image_folders:
        print(f'======================Processing {image_folder}')
        building=image_folder.split('_')[0]
        if building in scales:
            if not exists(join(root,'logs','unav',image_folder,'neighbors.h5')):
                extract_descriptors(join(root,'logs','unav',image_folder),teacher_model.model)
                gt=get_gt(join(root,'data','unav','utm',image_folder+'.json'))
                find_neighbors(join(root,'logs','unav',image_folder),gt,scales[image_folder.split('_')[0]],global_descriptor_dim,posDistThr,nonTrivPosDistSqThr,nPosSample)
            
if __name__=='__main__':
    device='cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/home/unav/Desktop/Resolution_Agnostic_VPR/configs/trainer.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        main(config)
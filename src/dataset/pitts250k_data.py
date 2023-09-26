import sys
from os.path import join,exists,basename
from os import listdir,makedirs
sys.path.append(join(sys.path[0],'..'))
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
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
import scipy.io
import shutil

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
    pitch_list=['000','030']
    yaw_list=[str(i*30).zfill(3) for i in range(12)]
    if not exists(hfile_path):
        hfile = h5py.File(hfile_path, 'a')
        grp = hfile.create_group(basename(image_folder))
        index=0
        for pitch in sorted(pitch_list):
            pitch_folder=join(image_folder,pitch)
            for ind,yaw in enumerate(sorted(yaw_list)):
                images_high_path=join(pitch_folder,yaw,'raw')
                image_high_names = set(sorted(listdir(images_high_path)))
                for i,im in tqdm(enumerate(image_high_names),desc=f'{str(ind).zfill(2)}/{len(yaw_list)}',total=len(image_high_names)):
                    if i%50==0 or i==len(image_high_names)-1:
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

def find_neighbors(image_folder,gt,global_descriptor_dim,posDistThr,nonTrivPosDistSqThr,nPosSample,query=False):
    if query:
        gt_name=join(config['root'],'data/third_party/pitts250k/groundtruth/tar/groundtruth/pittsburgh_queryID_1000.mat')
        name_id = scipy.io.loadmat(gt_name)['query_id'][0]
        name_id=[int(i)-1 for i in name_id]
    else:
        name_id=[i for i,_ in enumerate(gt)]

    pitch_list=['000','030']
    yaw_list=[str(i*30).zfill(3) for i in range(12)]

    hfile_path=join(image_folder,'global_descriptor.h5')
    hfile = h5py.File(hfile_path, 'r')
    
    names=[]
    descriptors=np.empty((len(hfile[basename(image_folder)]),global_descriptor_dim))
    locations=np.empty((len(hfile[basename(image_folder)]),2))

    for i,(k,v) in tqdm(enumerate(hfile[basename(image_folder)].items()),desc='load data',total=len(hfile[basename(image_folder)])):
        names.append(k)
        pitch,yaw,name=k.replace('.jpg','').split('+')
        locations[i,:]=gt[name_id.index(int(name))]
        descriptors[i,:]=v.__array__()

    knn = NearestNeighbors(n_jobs=1)
    knn.fit(gt)
    nontrivial_positives = list(knn.radius_neighbors(gt,
            radius=nonTrivPosDistSqThr, 
            return_distance=False))

    potential_positives = knn.radius_neighbors(gt,
            radius=posDistThr, 
            return_distance=False)
    potential_negatives = []
    for pos in potential_positives:
        potential_negatives.append(np.setdiff1d(np.arange(len(gt)),
            pos, assume_unique=True))

    descriptors=torch.from_numpy(descriptors)
    batch_size = 1000  # or any other batch size you find suitable

    # Open the HDF5 file
    hfile_neighbor_path = join(image_folder, 'neighbors.h5')
    hfile = h5py.File(hfile_neighbor_path, 'a')

    # Iterate through batches
    num_batches = int(np.ceil(len(names)/batch_size))

    for batch_idx in tqdm(range(num_batches), desc='find neighbor'):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(names))
        descriptor_batch = descriptors[start_idx:end_idx]
        # Compute the similarity matrix for the batch against all descriptors
        sim_batch = torch.einsum('id,jd->ij', descriptor_batch, descriptors).float()
        for i in range(descriptor_batch.shape[0]):
            sim_batch[i, start_idx + i] = 0
        
        for i in range(descriptor_batch.shape[0]):
            name = names[start_idx + i]
            sim = sim_batch[i]
            feature_closed = torch.topk(sim, descriptors.size(0), dim=0).indices.numpy()
            
            key = name.replace('.jpg', '').split('+')[-1]
            physical_closed = set([str(name_id[ind]).zfill(6) for ind in nontrivial_positives[name_id.index(int(key))]])
            
            negtives_ = [str(name_id[ind]).zfill(6) for ind in potential_negatives[name_id.index(int(key))]]
            negtives = []
            for pitch in pitch_list:
                for yaw in yaw_list:
                    for neg in negtives_:
                        negtives.append(f'{pitch}+{yaw}+{neg}.jpg')
            negtives = random.sample(negtives, 100)

            positives = []
            ind = 0
            while len(positives) < nPosSample * 20:
                key = names[feature_closed[ind]].replace('.jpg', '').split('+')[-1]
                if key in physical_closed:
                    positives.append(names[feature_closed[ind]])
                ind += 1
                if ind >= len(feature_closed):
                    break

            if len(positives) > 0:
                grp = hfile.create_group(name)
                grp.create_dataset('positives', data=positives)
                grp.create_dataset('negtives', data=negtives)

    hfile.close()

def process_data(root,database_gt):
    output_folder=join(root,'logs','pitts250k')
    types=['train','valid','database','query']
    pitch_list=['000','030']
    yaw_list=[str(i*30).zfill(3) for i in range(12)]
    resolutions={'raw':[640,480],'180p':[240,180],'360p':[480,360],'240p':[320,240]}
    for t in types:
        outf=join(output_folder,t)
        for PITCH in pitch_list:
            pitch_folder=join(outf,PITCH)
            for YAW in yaw_list:
                yaw_folder=join(pitch_folder,YAW)
                for k in list(resolutions.keys()):
                    image_folder=join(yaw_folder,k)
                    if not exists(image_folder):
                        makedirs(image_folder)
    resolutions.pop('raw')

    database_folders=[str(i).zfill(3) for i in range(11)]

    kmeans = KMeans(n_clusters=10).fit(database_gt)
    labels = kmeans.labels_
    train_indices, temp_indices, train_data, temp_data = train_test_split(
        np.arange(len(database_gt)), database_gt, test_size=0.3, stratify=labels
    )

    # Split temp_data into validation and test sets using stratified sampling based on temp_labels
    valid_indices, test_indices, valid_data, test_data = train_test_split(
        temp_indices, temp_data, test_size=0.7, stratify=labels[temp_indices]
    )

    input_folder=join(root,'data','third_party','pitts250k')
    for folder in database_folders:
        images=listdir(join(input_folder,folder))
        images.remove('tar')
        for image in tqdm(images,desc=f'process {folder}',total=len(images)):
            name,pitch,yaw=image.replace('.jpg','').split('_')
            name_=int(name)
            pitch=str((int(pitch.replace('pitch',''))-1)*30).zfill(3)
            yaw=str((int(yaw.replace('yaw',''))-1)*30).zfill(3)
            if name_ in train_indices:
                type='train'
            elif name_ in valid_indices:
                type='valid'
            elif name_ in test_indices:
                type='database'
            input_path=join(input_folder,folder,image)
            output_path=join(output_folder,type,pitch,yaw)
            shutil.copy(input_path,join(output_path,'raw',name+'.jpg'))
            image_=cv2.imread(input_path)
            for resolution,newsize in resolutions.items():
                image_new=cv2.resize(image_,newsize)
                cv2.imwrite(join(output_path,resolution,name+'.jpg'),image_new)

    query_folder=join(root,'data','third_party','pitts250k','queries_real')
    images=listdir(query_folder)
    images.remove('tar')
    for image in images:
        name,pitch,yaw=image.replace('.jpg','').split('_')
        name_=int(name)
        pitch=str((int(pitch.replace('pitch',''))-1)*30).zfill(3)
        yaw=str((int(yaw.replace('yaw',''))-1)*30).zfill(3)
        type='query'
        input_path=join(query_folder,image)
        output_path=join(output_folder,type,pitch,yaw)
        shutil.copy(input_path,join(output_path,'raw',name+'.jpg'))
        image_=cv2.imread(input_path)
        for resolution,newsize in resolutions.items():
            image_new=cv2.resize(image_,newsize)
            cv2.imwrite(join(output_path,resolution,name+'.jpg'),image_new)

def main(configs):
    root=configs['root']
    content=configs['vpr']['global_extractor']['netvlad']
    teacher_model=NetVladFeatureExtractor(join(configs['root'], content['ckpt_path']), arch=content['arch'],
        num_clusters=content['num_clusters'],
        pooling=content['pooling'], vladv2=content['vladv2'], nocuda=content['nocuda'])
    teacher_model.model.to(device).eval()

    gt_path=join(root,'data/third_party/pitts250k/groundtruth/tar/groundtruth/pittsburgh_database_10586_utm.mat')
    database_gt = scipy.io.loadmat(gt_path)['Cdb'].T

    gt_path=join(root,'data/third_party/pitts250k/groundtruth/tar/groundtruth/pittsburgh_query_1000_utm.mat')
    query_gt = scipy.io.loadmat(gt_path)['Cq'].T
    if not exists(join(root,'logs','pitts250k')):
        process_data(root,database_gt)

    global_descriptor_dim=configs['train']['num_cluster']*configs['train']['cluster']['dimension']
    posDistThr=configs['train']['triplet_loss']['posDistThr']
    nonTrivPosDistSqThr=configs['train']['triplet_loss']['nonTrivPosDistSqThr']
    nPosSample=configs['train']['triplet_loss']['nPosSample']

    for image_folder in ['train','valid','database','query']:
        print(f'======================Processing {image_folder}')
        if image_folder=='query':
            gt=query_gt
        else:
            gt=database_gt
        if not exists(join(root,'logs','pitts250k',image_folder,'neighbors.h5')):
            extract_descriptors(join(root,'logs','pitts250k',image_folder),teacher_model.model)
            if image_folder=='query':
                find_neighbors(join(root,'logs','pitts250k',image_folder),gt,global_descriptor_dim,posDistThr,nonTrivPosDistSqThr,nPosSample,query=True)
            else:
                find_neighbors(join(root,'logs','pitts250k',image_folder),gt,global_descriptor_dim,posDistThr,nonTrivPosDistSqThr,nPosSample)
            
if __name__=='__main__':
    device='cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/home/unav/Desktop/Resolution_Agnostic_VPR/configs/trainer.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        main(config)
# Import necessary libraries
import sys
from os.path import join, exists, basename
from os import listdir, makedirs
# Append the parent directory to sys.path to allow importing from there
sys.path.append(join(sys.path[0], '..'))
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
# Import NetVladFeatureExtractor from a third-party library
from third_party.pytorch_NetVlad.Feature_Extractor import NetVladFeatureExtractor
import argparse
import yaml
import json
import random
import scipy.io
import shutil

# Function to define input transformations for images
def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

# Function to extract descriptors from images using a global feature extractor
def extract_descriptors(image_folder, global_extractor):
    hfile_path = join(image_folder, 'global_descriptor.h5')
    if not exists(hfile_path):
        hfile = h5py.File(hfile_path, 'a')
        grp = hfile.create_group(basename(image_folder))
        index = 0
        
        images_high_path = join(image_folder, 'raw')
        image_high_names = set(sorted(listdir(images_high_path)))
        for i, im in tqdm(enumerate(image_high_names), total=len(image_high_names)):
            if i % 50 == 0 or i == len(image_high_names) - 1:
                if i > 0:
                    image_ = torch.stack(images_list)
                    feature = global_extractor.encoder(image_)
                    vector = global_extractor.pool(feature).detach().cpu()
                    for name, descriptor in zip(images_name, vector):
                        grp.create_dataset(name, data=descriptor)
                    index += vector.size(0)
                    del image_, feature, vector
                    torch.cuda.empty_cache()
                images_list = []
                images_name = []
            image = input_transform()(Image.open(join(images_high_path, im)))
            images_list.append(image.to(device))
            images_name.append(f'{im}')
        hfile.close()

# Function to find neighbors for each image based on their global descriptors
def find_neighbors(name_id, image_folder, gt, global_descriptor_dim, posDistThr, nonTrivPosDistSqThr, nPosSample, query=False):
    hfile_path = join(image_folder, 'global_descriptor.h5')
    hfile = h5py.File(hfile_path, 'r')
    
    names = []
    descriptors = np.empty((len(hfile[basename(image_folder)]), global_descriptor_dim))
    locations = np.empty((len(hfile[basename(image_folder)]), 2))

    for i, (k, v) in tqdm(enumerate(hfile[basename(image_folder)].items()), desc='load data', total=len(hfile[basename(image_folder)])):
        names.append(k)
        name = k.split('@')[7] # use pano_id as name
        locations[i, :] = gt[name_id.index(name)]
        descriptors[i, :] = v.__array__()

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

    descriptors = torch.from_numpy(descriptors)
    batch_size = 1000  # or any other batch size you find suitable

    # Open the HDF5 file for storing neighbors
    hfile_neighbor_path = join(image_folder, 'neighbors.h5')
    hfile = h5py.File(hfile_neighbor_path, 'a')

    # Iterate through batches to find neighbors
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
            
            key = name.split('@')[7]
            physical_closed = set([str(name_id[ind]).zfill(6) for ind in nontrivial_positives[name_id.index(key)]])
            
            negtives_ = [str(name_id[ind]).zfill(6) for ind in potential_negatives[name_id.index(key)]]
            negtives = [f'{neg}.jpg' for neg in negtives_]
            negtives = random.sample(negtives, 100)

            positives = []
            ind = 0
            while len(positives) < nPosSample * 20:
                key = names[feature_closed[ind]].replace('.jpg', '').split('@')[7]
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

# Function to process data, organizing it into the required structure for training and evaluation
def process_data(root, input_path, output_dir_name):
    output_folder = join(root, 'logs', output_dir_name)
    types = ['database', 'queries']
    resolutions = {'raw': [640, 480], '180p': [240, 180], '360p': [480, 360], '240p': [320, 240]}
    for t in types:
        outf = join(output_folder, t)
        for k in list(resolutions.keys()):
            image_folder = join(outf, k)
            if not exists(image_folder):
                makedirs(image_folder)
                
    # resolutions.pop('raw') # don't pop; copy over raw images to the corresponding raw folders
    
    for t in types:
        current_input_folder = join(input_path, t)
        current_output_folder = join(output_folder, t)
        images = [img for img in listdir(current_input_folder) if img.endswith('.jpg')]
        
        for image in tqdm(images, desc=f'Processing {t} images'):
            input_image_path = join(current_input_folder, image)
            image_ = cv2.imread(input_image_path)
            for resolution, newsize in resolutions.items():
                resized_image = cv2.resize(image_, tuple(newsize))
                output_image_path = join(current_output_folder, resolution, image)
                output_dir = join(current_output_folder, resolution)
                if not exists(output_dir):
                    makedirs(output_dir)
                cv2.imwrite(output_image_path, resized_image)

def process_image_filenames(folder_path):
    # Initialize lists to store UTM coordinates and panorama IDs
    utm_coords = []
    pano_ids = []
    
    # Check if the folder exists
    if not exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return None
    
    # Iterate through each file in the folder
    for filename in listdir(folder_path):
        if filename.endswith('.jpg'):
            # Split the filename to extract the required information
            parts = filename.split('@')
            if len(parts) >= 13:  # Ensure there are enough parts to extract data
                try:
                    # Extract UTM east, UTM north, and pano_id
                    utm_east = float(parts[1].strip())
                    utm_north = float(parts[2].strip())
                    pano_id = parts[7].strip()
                    
                    # Append the extracted information to the lists
                    utm_coords.append([utm_east, utm_north])
                    pano_ids.append(pano_id)
                except ValueError:
                    # If conversion to float fails, skip this file
                    print(f"Skipping file due to invalid format: {filename}")
    
    # Convert lists to numpy arrays
    utm_coords_array = np.array(utm_coords)
    
    # Return the 2D numpy array of UTM coordinates and 1D array of panorama IDs as a tuple
    return (utm_coords_array, pano_ids)

# Main function to run the entire script
def main(configs):
    root = configs['root']
    content = configs['vpr']['global_extractor']['netvlad']
    teacher_model = NetVladFeatureExtractor(join(configs['root'], content['ckpt_path']), arch=content['arch'],
                                            num_clusters=content['num_clusters'],
                                            pooling=content['pooling'], vladv2=content['vladv2'], nocuda=content['nocuda'])
    teacher_model.model.to(device).eval()
    
    # note: edit the yaml file and set root to your project root
    # also add a "name" field under the trainer_dataset.yaml's `data` field
    input_path = join(root, configs['data']['name'], 'images', 'test')
    
    database_input_path = join(input_path, 'database')
    database_gt, database_ids = process_image_filenames(database_input_path)
    
    query_input_path = join(input_path, 'queries')
    query_gt, query_ids = process_image_filenames(query_input_path)
    
    # note: this is the directory the output images and h5 files will be in under root
    logs_dir_name = "test_logs"
    output_dir_name = configs['data']['name']
    output_dir = join(root, logs_dir_name, output_dir_name)
    
    if not exists(output_dir):
        process_data(root, input_path, output_dir)

    global_descriptor_dim = configs['train']['num_cluster']*configs['train']['cluster']['dimension']
    posDistThr = configs['train']['triplet_loss']['posDistThr']
    nonTrivPosDistSqThr = configs['train']['triplet_loss']['nonTrivPosDistSqThr']
    nPosSample = configs['train']['triplet_loss']['nPosSample']

    for image_folder in ['database', 'queries']:
        print(f'======================Processing {image_folder}')
        if image_folder == 'queries':
            gt = query_gt
            ids = query_ids
        else:
            gt = database_gt
            ids = database_ids
            
        image_folder_path = join(root, logs_dir_name, configs['data']['name'], image_folder)
            
        if not exists(join(image_folder_path, 'neighbors.h5')):
            extract_descriptors(image_folder_path, teacher_model.model)
            if image_folder == 'queries':
                find_neighbors(ids, image_folder_path, gt, global_descriptor_dim, posDistThr, nonTrivPosDistSqThr, nPosSample, query=True)
            else:
                find_neighbors(ids, image_folder_path, gt, global_descriptor_dim, posDistThr, nonTrivPosDistSqThr, nPosSample)

# Check if the script is being run directly and, if so, execute the main function
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/scratch/zl3493/VPR4LQQ/configs/trainer_st_lucia.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        main(config)

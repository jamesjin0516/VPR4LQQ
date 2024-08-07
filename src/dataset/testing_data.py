# Import necessary libraries
from os.path import join, exists, basename, splitext
from os import listdir, makedirs, rename
import numpy as np
from sklearn.neighbors import NearestNeighbors
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import h5py
from feature.Global_Extractors import GlobalExtractors
import argparse
import yaml
import random


# Function to define input transformations for images
def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

# Function to extract descriptors from images using a global feature extractor
def extract_descriptors(image_folder, model):
    hfile_path = join(image_folder, f"global_descriptor_{model}.h5")
    hfile, grp_name = h5py.File(hfile_path, 'a'), basename(image_folder)
    # Retrieve already processed images, if any
    if grp_name in hfile:
        existing_imgs = set(hfile[grp_name])
        grp = hfile[grp_name]
    else:
        existing_imgs = set()
        grp = hfile.create_group(basename(image_folder))

    # Ignore already processed images
    images_high_path = join(image_folder, 'raw')
    image_high_names = set(listdir(images_high_path))
    images_to_add = image_high_names.difference(existing_imgs)

    if len(images_to_add) == 0: print(f"{basename(image_folder)} {basename(hfile_path)} already contains all images.")
    for i, im in tqdm(enumerate(images_to_add), desc=f"{basename(image_folder)} {basename(hfile_path)}", total=len(images_to_add)):
        image = Image.open(join(images_high_path, im))
        if image.mode != "RGB": image = image.convert("RGB")
        image = input_transform()(image)
        # If 5 images read, no more images to read, or image resolution changed, compute descriptors
        if i % 5 == 0 or i == len(images_to_add) - 1 or (len(images_list) > 0 and images_list[-1].shape != image.shape):
            if i > 0:
                batched_imgs = torch.stack(images_list)
                vector = global_extractors(model, batched_imgs)
                for name, descriptor in zip(images_name, vector):
                    grp.create_dataset(name, data=descriptor)
                del batched_imgs, vector
                torch.cuda.empty_cache()
            images_list, images_name = [], []
        images_list.append(image.to(device))
        images_name.append(im)
    if len(images_list) == 1: grp.create_dataset(im, data=global_extractors(model, image.to(device).unsqueeze(0)).squeeze(0))
    hfile.close()

# Function to find neighbors for each image based on their global descriptors
def find_neighbors(name_id, image_folder, model, gt, global_descriptor_dim, posDistThr, nonTrivPosDistSqThr, nPosSample):
    hfile_neighbor_path = join(image_folder, f"neighbors_{model}.h5")
    hfile_path = join(image_folder, f"global_descriptor_{model}.h5")
    hfile = h5py.File(hfile_path, 'r')
    
    names = []
    descriptors = np.empty((len(hfile[basename(image_folder)]), global_descriptor_dim))
    locations = np.empty((len(hfile[basename(image_folder)]), 2))

    for i, (img_name, img_feat) in tqdm(enumerate(hfile[basename(image_folder)].items()), desc=f"Loading {basename(image_folder)} {basename(hfile_path)}", total=len(hfile[basename(image_folder)])):
        names.append(img_name)
        locations[i, :] = gt[name_id.index(img_name)]
        descriptors[i, :] = img_feat.__array__()
    hfile.close()

    knn = NearestNeighbors(n_jobs=-1)
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
    hfile = h5py.File(hfile_neighbor_path, 'a')
    existing_indices = set(names.index(name) for name in set(hfile))
    indices_to_add = set(range(len(names))).difference(existing_indices)

    # Iterate through batches to find neighbors
    num_batches = int(np.ceil(len(names)/batch_size))
    
    if len(indices_to_add) == 0: print(f"{basename(image_folder)} {basename(hfile_neighbor_path)} already contains all images.")
    for batch_idx in tqdm(range(num_batches), desc=f"{basename(image_folder)} {basename(hfile_neighbor_path)}"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(names))
        descriptor_batch = descriptors[start_idx:end_idx]
        # Compute the similarity matrix for the batch against all descriptors
        sim_batch = torch.einsum('id,jd->ij', descriptor_batch, descriptors).float()
        for i in range(descriptor_batch.shape[0]):
            sim_batch[i, start_idx + i] = 0
        
        # Between start and end index, keep only uncalculated indices (ie. the corresponding image isn't in neighbors file)
        for i in [ind - start_idx for ind in indices_to_add.intersection(range(start_idx, end_idx))]:
            name = names[start_idx + i]
            sim = sim_batch[i]
            feature_closed = torch.topk(sim, descriptors.size(0), dim=0).indices.numpy()
            
            physical_closed = set([str(name_id[ind]).zfill(6) for ind in nontrivial_positives[name_id.index(name)]])
            
            negtives = [str(name_id[ind]).zfill(6) for ind in potential_negatives[name_id.index(name)]]
            negtives = random.sample(negtives, 100)

            positives = []
            ind = 0
            while len(positives) < nPosSample * 20:
                key = names[feature_closed[ind]]
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
def process_data(database_path, query_path, resolutions, img_ext):
    split_types = {"database": database_path, "query": query_path}
    for split_type, split_path in split_types.items():
        # Move any unclassifed images under database or query to the "raw" resolution folder
        images = [image for image in listdir(split_path) if splitext(image)[1].replace('.', '') == img_ext]
        raw_folder = join(split_path, "raw")
        if not exists(raw_folder):
            makedirs(raw_folder)
        for image in images:
            rename(join(split_path, image), join(raw_folder, image))
        if split_type == "database": continue
        # Ensure all other resolutions have a cooresponding folder
        for res_name in list(resolutions.keys()):
            image_folder = join(split_path, res_name)
            if not exists(image_folder):
                makedirs(image_folder)

        # Resize and copy all images moved to raw folder into corresponding resolution folders
        for image in tqdm(images, desc=f'Processing {split_type} images'):
            raw_image_path = join(raw_folder, image)
            image_ = cv2.imread(raw_image_path)
            for resolution, newsize in resolutions.items():
                if isinstance(newsize, int):
                    newsize = (newsize, int(image_.shape[1] * newsize / image_.shape[0]))
                resized_image = cv2.resize(image_, tuple(reversed(newsize)))    # cv2.resize expects (width, height)
                output_image_path = join(split_path, resolution, image)
                cv2.imwrite(output_image_path, resized_image)


def process_image_filenames(folder_path, img_ext):
    # Initialize lists to store UTM coordinates and panorama IDs
    utm_coords = []
    filenames = []
    
    # Check if the folder exists
    if not exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return None
    
    # Iterate through each file in the folder
    for filename in listdir(folder_path):
        ext = splitext(filename)[1].replace('.', '')
        if ext == img_ext:
            # Split the filename to extract the required information
            parts = filename.split('@')
            try:
                # Assume the x and y coordinates of image ground truth is 1st and 2nd component
                utm_east = float(parts[1].strip())
                utm_north = float(parts[2].strip())
                
                # Append the extracted information to the lists
                utm_coords.append([utm_east, utm_north])
                filenames.append(filename)
            except (ValueError, AssertionError):
                # If conversion to float fails, skip this file
                print(f"Skipping file due to invalid format: {filename}")
    
    # Convert lists to numpy arrays
    utm_coords_array = np.array(utm_coords)
    
    # Return the 2D numpy array of UTM coordinates and 1D array of panorama IDs as a tuple
    return (utm_coords_array, filenames)

# Main function to run the entire script
def main(configs, data_info):
    # Assemble the path to where the testing dataset contains database and query folders
    testset = configs['test_data']['name']
    test_info = data_info[testset]
    testset_path = join(configs['root'], data_info["testsets_path"], testset, test_info["subset"])
    img_ext = test_info["img_ext"]
    
    # Divide images into resolutions for each of database and query folders
    database_input_path, query_input_path = join(testset_path, test_info["database"]), join(testset_path, test_info["query"])
    process_data(database_input_path, query_input_path, configs["test_data"]["resolution"], img_ext)

    # Parse coordinates from all images directly under the database and query folders of the dataset
    gt_info = {"database": dict(zip(["gt", "filenames"], process_image_filenames(join(database_input_path, "raw"), img_ext))),
               "query": dict(zip(["gt", "filenames"], process_image_filenames(join(query_input_path, "raw"), img_ext)))}
    
    posDistThr = configs['train_conf']['triplet_loss']['posDistThr']
    nonTrivPosDistSqThr = configs['train_conf']['triplet_loss']['nonTrivPosDistSqThr']
    nPosSample = configs['train_conf']['triplet_loss']['nPosSample']

    # Generate permutations of image types and vpr methods
    arguments = []
    for image_type in gt_info:
        arguments.extend((image_type, model) for model in global_extractors.models)
    # Expand permutations into global descriptor extraction parameters
    descriptor_args = [(join(testset_path, test_info[image_folder]), model) for image_folder, model in arguments]
    print(f"Extracting global descriptors of {testset} for each of {global_extractors.models}")
    for arg in descriptor_args:
        extract_descriptors(*arg)
    # Similarly assemble find neighbor parameters
    neighbor_args = [(gt_info[image_folder]["filenames"], join(testset_path, test_info[image_folder]), model, gt_info[image_folder]["gt"],
                      global_extractors.feature_length(model), posDistThr, nonTrivPosDistSqThr, nPosSample) for image_folder, model in arguments]
    print(f"Gathering neighbors of {testset} for each of {global_extractors.models}")
    for arg in neighbor_args:
        find_neighbors(*arg)

# Check if the script is being run directly and, if so, execute the main function
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../configs/test_trained_model.yaml')
    parser.add_argument("-d", "--data_info", type=str, default="../configs/testing_data.yaml")
    parser.add_argument("--test_set", type=str, help="Name of dataset to use")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        configs = yaml.safe_load(f)
    with open(args.data_info, "r") as d_locs_file:
        data_info = yaml.safe_load(d_locs_file)
    if args.test_set is not None:
        configs["test_data"]["name"] = args.test_set
    global_extractors = GlobalExtractors(configs["root"], configs["vpr"]["global_extractor"], preprocess=True)
    main(configs, data_info)

import argparse

import yaml
import faiss
import h5py
import numpy as np
from os.path import basename, join
from sklearn.neighbors import NearestNeighbors
from tensorboardX import SummaryWriter
import torch
from tqdm import tqdm

from tools.GoogleSheet import GoogleSheet


# !!!!!!! DO NOT MODIFY DATASET LISTS !!!!!!!    (order determines Google Sheet write location)
SMALL_DATASETS = ["st_lucia", "amstertime", "tum_lsi", "sped"]    
MEDIUM_DATASETS = ["msls", "Nordland", "GangnamStation", "nyc_indoor"]
LARGE_DATASETS = ["tokyo247", "eynsham", "svox"]

SHEET_START_COL = "C"
SHEET_START_ROW = 3


def get_data_index(data_name):
    if data_name in SMALL_DATASETS:
        return SMALL_DATASETS.index(data_name)
    if data_name in MEDIUM_DATASETS:
        return len(SMALL_DATASETS) + MEDIUM_DATASETS.index(data_name)
    return len(SMALL_DATASETS) + len(MEDIUM_DATASETS) + LARGE_DATASETS.index(data_name)


def global_descriptors_paths(data_conf, data_folders, train_conf, models_to_test):
    train_data = train_conf["data"]["name"]
    if not data_conf["use_trained_descs"]:
        log_suffix = "pretrained" 
    elif train_conf["finetuned"]:
        log_suffix = f"{train_data}_finetuned"
    else:
        log_suffix = join(train_data + ("_distill" if train_conf["loss"]["distill"] else "") + ("_vlad" if train_conf["loss"]["vlad"] else "") \
                        + ("_triplet" if train_conf["loss"]["triplet"] else ""),
                        str(train_conf["data"]["resolution"]) + "_" + str(train_conf["data"]["qp"]) + "_" + str(train_conf["lr"]))
    desc_file_paths = {}
    for model in models_to_test:
        desc_filename_formatter = f"global_descriptor_{{}}.h5" if data_conf["test_res"] == "raw" else f"global_descriptor_{{}}_{data_conf['test_res']}.h5"
        desc_file_paths[model] = {img_type: join(data_folders[img_type], "precomputed_descriptors", log_suffix, desc_filename_formatter.format(model))
                                     for img_type in data_folders}
    return desc_file_paths, log_suffix


def load_all_descriptors(hfile_path, image_folder):
    g_desc_file = h5py.File(hfile_path, "r")
    num_images = len(g_desc_file[basename(image_folder)])
    locations = torch.empty((num_images, 2))
    descriptors = []
    names = []
    for i, (name, d) in enumerate(tqdm(g_desc_file[basename(image_folder)].items(), desc=f"Reading database info for {basename(image_folder)}")):
        splitted = name.split('@')
        utm_east_str, utm_north_str = splitted[1], splitted[2]
        image_path = join(image_folder, 'raw', name)
        names.append(image_path)
        locations[i] = torch.tensor([float(utm_east_str), float(utm_north_str)], dtype=torch.float32)
        descriptors.append(d.__array__())
    g_desc_file.close()
    descriptors = torch.tensor(np.vstack(descriptors))
    return descriptors, locations


def calculate_recall(database_features, queries_features, database_utm, query_utm, recall_values, threshold):
    # Find soft_positives_per_query, which are within val_positive_dist_threshold (deafult 25 meters)
    knn = NearestNeighbors(n_jobs=-1)
    knn.fit(database_utm)
    soft_positives_per_query = knn.radius_neighbors(query_utm, radius=threshold, return_distance=False)
    
    faiss_index = faiss.IndexFlatL2(database_features.shape[1])
    faiss_index.add(database_features)
    del database_features
    
    distances, predictions = faiss_index.search(queries_features, max(recall_values).item())

    #### For each query, check if the predictions are correct
    positives_per_query = soft_positives_per_query
    # args.recall_values by default is [1, 5, 10, 20]
    recalls = np.zeros(len(recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / len(queries_features) * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(recall_values, recalls)])
    return recalls, recalls_str


def write_results(model_name, dataset_name, tensorboard, googlesheet, topk_nodes, recall_rate, sheet_col_offset):
    dataset_ind = get_data_index(dataset_name)
    for i, topk in enumerate(topk_nodes):
        tensorboard.add_scalar(f"{model_name}/Recall rate/@{int(topk)}", recall_rate[i])
    dset_name_loc = googlesheet.column_shift(SHEET_START_COL, -2) + str(SHEET_START_ROW + dataset_ind * len(topk_nodes))
    googlesheet.write_cell(model_name, dset_name_loc, dataset_name)
    
    topk_col = googlesheet.column_shift(SHEET_START_COL, -1)
    loss_col = googlesheet.column_shift(SHEET_START_COL, sheet_col_offset)

    for topk_ind, topk_score in enumerate(recall_rate.tolist()):
        dest_row = str(SHEET_START_ROW + dataset_ind * len(topk_nodes) + topk_ind)
        googlesheet.write_cell(model_name, loss_col + dest_row, topk_score)
        googlesheet.write_cell(model_name, topk_col + dest_row, f"R@{topk_nodes[topk_ind]}")


def test_vpr(root, data_conf, data_folders, vpr_conf, train_conf, model_IO, sheet_col_offset):
    models_to_test = [model_name for model_name in vpr_conf["global_extractor"] if vpr_conf["global_extractor"][model_name]["use"]]
    desc_file_paths, log_suffix = global_descriptors_paths(data_conf, data_folders, train_conf, models_to_test)
    print(f"{data_conf['name']} ({data_conf['test_res']}) testing with {log_suffix} {models_to_test}")
    log_dir = join(root, model_IO["logs_path"], data_conf["name"] + "_test", "res_" + data_conf["test_res"], log_suffix)
    tensorboard = SummaryWriter(log_dir=log_dir)
    googlesheet = GoogleSheet("VPR4LQQ ICRA Distillation Results (Public Code)")
    topk_nodes = torch.tensor(vpr_conf["topk"])

    for model_name, file_paths in desc_file_paths.items():
        database_descs, database_utm =  load_all_descriptors(file_paths["database"], data_folders["database"])
        query_descs, query_utm =  load_all_descriptors(file_paths["query"], data_folders["query"])
        recall_rate, recall_str = calculate_recall(database_descs, query_descs, database_utm, query_utm, topk_nodes, vpr_conf["threshold"][0])
        print(f"{model_name} recall rates on {data_conf['name']}: {recall_str}")
        write_results(model_name, data_conf["name"], tensorboard, googlesheet, topk_nodes, recall_rate, sheet_col_offset)


def main(configs, data_info):
    root = configs['root']
    data_name = configs['test_data']['name']

    test_data_dir_name = data_info['testsets_path']
    database_folder = data_info[data_name]['database']
    query_folder = data_info[data_name]['query']
    subset = data_info[data_name]['subset']

    data_folders = {
        "database": join(root, test_data_dir_name, data_name, subset, database_folder),
        "query": join(root, test_data_dir_name, data_name, subset, query_folder)
    }
    if "valid" in data_info[data_name]: data_folders["valid"] = join(root, test_data_dir_name, data_name, subset, data_info["valid"])
    test_vpr(root, configs['test_data'], data_folders, configs['vpr'], configs['train_conf'], configs["model_IO"], configs["sheet_col_offset"])


def update_config_with_args(args, config):
    if args.test_set is not None:
        config["test_data"]["name"] = args.test_set
    if args.config_num is not None:
        config["test_data"]["test_res"] = "raw" if args.config_num == 1 else "90%"
        config["test_data"]["use_trained_descs"] = args.config_num > 2
        config["train_conf"]["finetuned"] = args.config_num == 3
        if args.config_num > 3:
            loss_combs = [(True, True, True), (True, True, False), (True, False, True), (True, False, False), (False, True, True), (False, True, False), (False, False, True)]
            for enabled, loss_type in zip(loss_combs[args.config_num - 4], config["train_conf"]["loss"]):
                config["train_conf"]["loss"][loss_type] = enabled
        config["sheet_col_offset"] = args.config_num - 1
    if args.models_to_test is not None:
        models = {1: "NetVlad", 2: "MixVPR", 3: "AnyLoc", 4: "DinoV2Salad", 5: "CricaVPR"}
        for model_num, model_name in models.items():
            config["vpr"]["global_extractor"][model_name]["use"] = model_num in args.models_to_test


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../configs/test_trained_model.yaml')
    parser.add_argument("-d", "--data_info", type=str, default="../configs/testing_data.yaml")
    parser.add_argument("--test_set", type=str, help="Name of dataset to use")
    parser.add_argument("--dset-group", type=int, help="If set, loop over 1 of 3 predefined dataset groups (1: small; 2: medium; 3: large)")
    parser.add_argument("--dset-sub-ind", type=int, help="If --dset-group is specified, further choose a specific dataset (for slurm array job)")
    parser.add_argument("--config-num", type=int, help="1: pretrained raw; 2: pretrained low; 3: finetuned; 4-10: loss combinations 1-7")
    parser.add_argument("--models-to-test", nargs="+", type=int, help="Optionally specify indicies of models tested, otherwise deduce from config file")
    args = parser.parse_args()

    with open(args.data_info, "r") as d_locs_file:
        data_info = yaml.safe_load(d_locs_file)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    if args.dset_group is not None and args.test_set is None:
        datasets = SMALL_DATASETS if args.dset_group == 1 else (MEDIUM_DATASETS if args.dset_group == 2 else LARGE_DATASETS)
        if args.dset_sub_ind is not None and args.dset_sub_ind != -1:
            update_config_with_args(args, config)
            config["test_data"]["name"] = datasets[args.dset_sub_ind]
            main(config, data_info)
        else:
            for dataset in datasets:
                with open(args.config, 'r') as f:
                    config = yaml.safe_load(f)
                update_config_with_args(args, config)
                config["test_data"]["name"] = dataset
                try:
                    main(config, data_info)
                except Exception as e:
                    print(f"!!!!!! {dataset} testing failed")
                    print(e)
    else:
        update_config_with_args(args, config)
        main(config, data_info)
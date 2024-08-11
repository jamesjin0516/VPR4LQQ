import argparse
import os
from os.path import basename, join
import random
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import torch
import h5py

from dataset.Data_Control import BatchSampler
from dataset.test_dataset import TestDataset, collate_fn
from feature.Global_Extractors import GlobalExtractors


class VPRTester:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, root, data_conf, data_folders, vpr_conf, train_conf, model_IO):
        self.extrs_pretrained = GlobalExtractors(root, vpr_conf["global_extractor"])
        # Assemble logs and weights folder name using training data name and loss and data configurations
        train_data = train_conf["data"]["name"]
        log_suffix = join(train_data + ("_distill" if train_conf["loss"]["distill"] else "") + ("_vlad" if train_conf["loss"]["vlad"] else "") \
                          + ("_triplet" if train_conf["loss"]["triplet"] else ""), "{}",
                          str(train_conf["data"]["resolution"]) + "_" + str(train_conf["data"]["qp"]) + "_" + str(train_conf["lr"]))

        for model_name in self.extrs_pretrained.models:
            vpr_conf["global_extractor"][model_name]["ckpt_path"] = join(root, model_IO["weights_path"], log_suffix.format(model_name))
        # Load the trained model, picking "model_best.pth" as weights via type="pipeline"
        self.extrs_trained = GlobalExtractors(root, vpr_conf["global_extractor"], pipeline=True)
        
        log_suffix = join(*[folder_name for folder_name in log_suffix.split(os.path.sep) if folder_name != "{}"])
        log_dir = join(root, model_IO["logs_path"], data_conf["name"] + "_test", "res_" + data_conf["test_res"], log_suffix)
        self.tensorboard = SummaryWriter(log_dir=log_dir)
        self.thresholds = torch.tensor(vpr_conf["threshold"])
        self.topk_nodes = torch.tensor(vpr_conf["topk"])
        self.batch_size = train_conf["batch_size"]
        
        self.train_conf = train_conf
        pretrained_suffix, trained_suffix = join("precomputed_descriptors", "pretrained"), join("precomputed_descriptors", log_suffix)
        self.database_images_set = self.load_database_images(data_folders, pretrained_suffix, trained_suffix)
        self.query_images_set = TestDataset(data_folders["query"], join(pretrained_suffix, f"neighbors_{self.extrs_pretrained.models[0]}.h5"),
                                            data_conf["test_res"], train_conf['triplet_loss'])

    def load_database_images(self, data_folders, pretrained_suffix, trained_suffix):
        image_folder = data_folders["database"]

        # Create descriptor file handles and empty descriptor tensors
        g_desc_files = {f"pretrained_{model}": h5py.File(join(image_folder, pretrained_suffix, f"global_descriptor_{model}.h5"), 'r') for model in self.extrs_pretrained.models}
        g_desc_files.update({model: h5py.File(join(image_folder, trained_suffix, f"global_descriptor_{model}.h5"), 'r') for model in self.extrs_trained.models})
        num_images = len(g_desc_files[list(g_desc_files.keys())[0]][basename(image_folder)])
        assert all([num_images == len(file[basename(image_folder)]) for file in g_desc_files.values()]), f"Database global descriptors for {image_folder} have different lengths (first file has length {num_images})."
        descriptors = {model: torch.empty((num_images, self.extrs_trained.feature_length(model))) for model in self.extrs_trained.models}
        for model, desc_mat in descriptors.items(): descriptors[model] = {"pretrained": desc_mat.clone(), "trained": desc_mat}
        locations = torch.empty((num_images, 2))
        names = []
        for file_ind, (model, g_desc_file) in enumerate(g_desc_files.items()):
            model_name, version = model.lstrip("pretrained_"), "pretrained" if model.startswith("pretrained_") else "trained"
            for i, (name, d) in enumerate(tqdm(g_desc_file[basename(image_folder)].items(), desc=f"Reading database info for {model}")):
                if file_ind == 0:
                    splitted = name.split('@')
                    utm_east_str, utm_north_str = splitted[1], splitted[2]
                    image_path = join(image_folder, 'raw', name)
                    names.append(image_path)
                    locations[i] = torch.tensor([float(utm_east_str), float(utm_north_str)], dtype=torch.float32)
                descriptors[model_name][version][i] = torch.tensor(d.__array__())
            g_desc_file.close()

        return {'images_path': names, 'descriptors': descriptors, 'locations': locations}
    
    def vpr_examing(self, query_desc, database_desc, query_loc, database_loc):
        query_desc, database_desc, query_loc, database_loc = query_desc.to(self.device), database_desc.to(self.device), query_loc.float().to(self.device), database_loc.float().to(self.device)
        sim = torch.einsum('id,jd->ij', query_desc, database_desc)
        topk_ = torch.topk(sim, self.topk_nodes[-1], dim=1)
        topk=topk_.indices
        success_num = torch.zeros((self.thresholds.shape[0],self.topk_nodes.shape[0]))
        for i, index in enumerate(topk):
            qloc = query_loc[i].unsqueeze(0)[:, :2]
            dloc = database_loc[index][:, :2]

            distance = torch.cdist(qloc, dloc, p=2).squeeze(0)

            matched = False
            for thre_ind,thre in enumerate(self.thresholds):
                if not matched:
                    if thre_ind==0:
                        ind = torch.where(distance < thre)
                    else:
                        ind=torch.where((distance < thre)&(distance>=self.thresholds[thre_ind-1]))
                    index_match=index[ind]
                    if len(index_match)>0:
                    # ind=torch.stack(list(ind),dim=0).squeeze(0).detach().cpu()
                    # if len(ind) > 0:
                    #     train_image_paths = [self.images_high_path[indd] for indd in index_match]
                    #     for index_, index_match_,train_image_path in zip(ind,index_match, train_image_paths):
                    #         if self.geometric_verification.geometric_verification(train_image_path, test_image_path):
                        maskA = torch.zeros((self.thresholds.shape[0], self.topk_nodes.shape[0]), dtype=torch.bool)
                        maskB = torch.zeros((self.thresholds.shape[0], self.topk_nodes.shape[0]), dtype=torch.bool)
                        maskA[self.thresholds >= thre, :] = True
                        maskB[:, self.topk_nodes > torch.min(ind[0]).item()] = True
                        mask = (maskA & maskB)
                        success_num[mask] += 1
                        matched = True


            del qloc,dloc,distance
        del sim,topk,query_desc,database_desc, query_loc, database_loc
        torch.cuda.empty_cache()
        return success_num

    def validation(self, iter_num):
        print(f"Start vpr examing #{iter_num} ...")
        
        # All query images are evaluated for VPR recall
        query_batch = 20
        batch_sampler = BatchSampler([list(range(len(self.query_images_set)))], query_batch)
        data_loader = DataLoader(self.query_images_set, collate_fn=collate_fn, num_workers=self.train_conf['num_worker'], pin_memory=True, batch_sampler=batch_sampler)

        recall_score = {model: {version: torch.zeros((self.thresholds.shape[0],self.topk_nodes.shape[0])) for version in ["pretrained", "trained"]} for model in self.extrs_trained.models}
        with torch.no_grad():
            for images_high, images_low, locations in tqdm(data_loader, total=len(self.query_images_set) // query_batch + 1):
                # For each image batch, apply pretrained and trained versions of each extractor and calculate separate recall rates
                locations = locations[:, 0, :]
                features_low, vectors_low = self._compute_image_descriptors(images_low, recall_score.keys())
                for model, vector_low in vectors_low.items():
                    for version in ["pretrained", "trained"]:
                        recall_score[model][version] += self.vpr_examing(vector_low[version], self.database_images_set["descriptors"][model][version], locations, self.database_images_set["locations"])
                del images_high, images_low, locations, features_low, vectors_low

            recall_rate = {model: {version: recall_score[model][version] / len(self.query_images_set) for version in["pretrained", "trained"]} for model in recall_score}

            for model, rate in recall_rate.items():
                for i,topk in enumerate(self.topk_nodes):
                    self.tensorboard.add_scalars(f"{model}/Recall rate/@{int(topk)}", {version: rate[version][0, i] for version in rate}, iter_num)
            
            del recall_score, recall_rate
        torch.cuda.empty_cache()

        if not hasattr(self, "valid_data_sets"):
            print("Not testing with loss testing data and / or trained models, exiting vpr examining.")
            return
        print("Start loss validation ...")
        random.seed(10)
        sample_size = 15
        len_init = len(self.valid_data_sets[0])
        indices = [random.sample(list(range(len_init)),sample_size)]
        for d in self.valid_data_sets[1:]:
            len_current = len_init + len(d)
            indices.append(random.sample(list(range(len_init,len_current)), sample_size))
            len_init = len_current
        batch_sampler = BatchSampler(indices, self.batch_size)
        data_loader = DataLoader(self.valid_data_set, num_workers=self.train_conf['num_worker'], pin_memory=True, batch_sampler=batch_sampler)

        with torch.no_grad():
            enabled_loss_types = set(self.train_conf["loss"].keys()).union(["loss"])
            models_losses = {model: {loss_type: 0 for loss_type in enabled_loss_types} for model in self.extrs_trained.models}
            for images_high, images_low, _ in tqdm(data_loader, total=len(self.valid_data_set) // self.batch_size + 1):
                features_high, vectors_high = self._compute_image_descriptors(images_high, models_losses.keys(), calc_pos_neg=True)
                features_low, vectors_low = self._compute_image_descriptors(images_low, models_losses.keys(), calc_pos_neg=True)
                B, G = len(images_high), images_high.shape[1] if isinstance(images_high, torch.Tensor) else len(images_high[0])
                with torch.autocast('cuda', torch.float32):
                    for model, losses in models_losses.items():
                        loss_sp = self.similarity_loss(features_low[model]["trained"], features_high[model]["trained"]).detach().cpu() / 1000
                        loss_vlad = self.vlad_mse_loss(vectors_low[model]["trained"], vectors_high[model]["trained"]).detach().cpu() * B * 100
                        loss_triplet, vector_low = 0, vectors_low[model]["trained"].view(B, G, -1)
                        for pos_neg_vecs in vector_low:
                            vladQ, vladP, vladN = torch.split(pos_neg_vecs, [1, self.nPosSample, self.nNegSample])
                            vladQ = vladQ.squeeze()
                            for vP in vladP:
                                for vN in vladN:
                                    loss_triplet += self.triplet_loss(vladQ, vP, vN)
                        loss_triplet /= (self.nPosSample * self.nNegSample * 10)
                        loss_triplet = loss_triplet.detach().cpu()
                        if "distill" in enabled_loss_types:
                            losses["distill"] += loss_sp
                        if "vlad" in enabled_loss_types:
                            losses["vlad"] += loss_vlad
                        if "triplet" in enabled_loss_types:
                            losses["triplet"] += loss_triplet
                        losses["loss"] += loss_sp + loss_vlad + loss_triplet

                del images_high, images_low, features_high, features_low, vectors_high, vectors_low
                torch.cuda.empty_cache()

            for model, losses in models_losses.items():
                for loss_type, loss_value in losses:
                    self.tensorboard.add_scalar(f"{model}/Valid/{'Loss' if loss_type == 'loss' else f'Loss_{loss_type}'}", loss_value / int(sample_size * len(self.valid_data_sets)), iter_num)
    
    def _compute_image_descriptors(self, input_images, models, calc_pos_neg=False):
        # Batches containing identical resolution images are stacked into tensors
        if isinstance(input_images, torch.Tensor):
            images = (input_images[:, :, 0, :, :, :] if calc_pos_neg else input_images[:, 0, 0, :, :, :]).to(self.device)
        # otherwise batches contain lists, which each have the query image and its positive & negative images
        else:
            images = []
            for batch_item in input_images:
                images.extend([batch_item[i].to(self.device) for i in range(len(batch_item) if calc_pos_neg else 1)])
        # Compute encoded features and global descriptors for images
        with torch.autocast('cuda', torch.float32):
            if isinstance(images, torch.Tensor):
                outputs = {model: {"pretrained": self.extrs_pretrained(model, images), "trained": self.extrs_trained(model, images)} for model in models}
            else:
                # Given different query image resolutions, compute sequentially
                pretrained_output = {model: [self.extrs_pretrained(model, image) for image in images] for model in models}
                trained_output = {model: [self.extrs_trained(model, image) for image in images] for model in models}
                outputs = {model: {"pretrained": [[img_out[0] for img_out in pretrained_output[model]], torch.cat([img_out[1] for img_out in pretrained_output[model]])],
                                   "trained": [[img_out[0] for img_out in trained_output[model]], torch.cat([img_out[1] for img_out in trained_output[model]])]} for model in models}
        encodings, descriptors = [{model: {version: data[i] for version, data in outputs[model].items()} for model in outputs} for i in range(2)]
        return encodings, descriptors


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
    
    vpr = VPRTester(root, configs['test_data'], data_folders, configs['vpr'], configs['train_conf'], configs["model_IO"])

    for iter_num in range(configs["begin_run"], configs["test_runs"] + 1):
        vpr.validation(iter_num)
    vpr.tensorboard.close()
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../configs/test_trained_model.yaml')
    parser.add_argument("-d", "--data_info", type=str, default="../configs/testing_data.yaml")
    parser.add_argument("--begin_run", type=int, help="Index for the initial vpr testing run.")
    parser.add_argument("--test_set", type=str, help="Name of dataset to use")
    parser.add_argument("--distill", type=bool, help="Enable or disable distill loss.")
    parser.add_argument("--vlad", type=bool, help="Enable or disable VLAD loss.")
    parser.add_argument("--triplet", type=bool, help="Enable or disable triplet loss.")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    if args.begin_run is not None:
        config["begin_run"] = args.begin_run
    if args.test_set is not None:
        config["test_data"]["name"] = args.test_set
    if args.distill is not None:
        config['train_conf']["loss"]["distill"] = args.distill
    if args.vlad is not None:
        config['train_conf']["loss"]["vlad"] = args.vlad
    if args.triplet is not None:
        config['train_conf']["loss"]["triplet"] = args.triplet

    with open(args.data_info, "r") as d_locs_file:
        data_info = yaml.safe_load(d_locs_file)
    main(config, data_info)

import random
import shutil
import h5py
import scipy
import torch.nn as nn
import yaml
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from os.path import basename, join
from os import makedirs

from dataset.Data_Control import collate_fn, collate_fn_gsv, load_gsv_cities_data, load_pitts250k_data, BatchSampler
from dataset.gsv_cities_data import CITIES_300x400, CITIES_480x640

from loss.loss_distill import Loss_distill
from feature.Global_Extractors import GlobalExtractors
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import ConcatDataset
import torch
import torch.backends.cuda as cuda
import torch.backends.cudnn as cudnn
import torch.backends.opt_einsum as opt_einsum

cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True
cudnn.benchmark = True
cuda.matmul.allow_fp16_reduced_precision_reduction = True
cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.set_default_dtype(torch.float32)
opt_einsum.enabled = True
cudnn.enabled = True


class VPR():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self, configs, data_info, data_name):
        train_conf, self.train_conf = configs["train"], configs["train"]
        extractors_conf = configs["vpr"]["global_extractor"]
        self.batch_size = train_conf["batch_size"]
        self.lr = train_conf["lr"]
        self.lr_decay = train_conf["lr_decay"]
        self.gamma = train_conf["exponential_gamma"]
        self.thresholds = torch.tensor(configs['vpr']['threshold'])
        self.topk_nodes = torch.tensor(configs['vpr']['topk'])

        self.teacher_models = GlobalExtractors(configs["root"], extractors_conf)

        logs_root = join(configs["root"], configs["model_IO"]["logs_path"], data_name)
        makedirs(logs_root, exist_ok=True)
        weights_root = join(configs["root"], configs["model_IO"]["weights_path"])
        self.weight_paths, self.log_dirs = {}, {}
        log_suffix = join(data_name + ("_distill" if train_conf["loss"]["distill"] else "") + ("_vlad" if train_conf["loss"]["vlad"] else "") \
                          + ("_triplet" if train_conf["loss"]["triplet"] else ""), "{}",
                          f"{train_conf['data']['resolution']}_{train_conf['data']['qp']}_{self.lr}")

        for model_name in self.teacher_models.models:
            model_suffix = log_suffix.format(model_name)
            self.weight_paths[model_name] = join(weights_root, model_suffix)
            self.log_dirs[model_name] = join(logs_root, model_suffix)
            makedirs(self.weight_paths[model_name], exist_ok=True)
            if train_conf["resume"]:
                extractors_conf[model_name]["ckpt_path"] = self.weight_paths[model_name]
        extractors_conf["MixVPR"]["img_size"] = configs["data"]["compression"]["resolution"][train_conf["data"]["resolution"]]

        self.log_writers = {model: SummaryWriter(log_dir=log_dir) for model, log_dir in self.log_dirs.items()}
        self.student_models = GlobalExtractors(configs["root"], extractors_conf, pipeline=train_conf["resume"])
        self.teacher_models.set_float32()
        self.student_models.set_float32()

        self.start_epochs = {model: self.student_models.last_epoch(model) + 1 if train_conf["resume"] else 0 for model in self.student_models.models}
        self.best_scores = {model: self.student_models.best_score(model) if train_conf["resume"] else 0 for model in self.student_models.models}
        triplet_loss_config = train_conf["triplet_loss"]
        self.nPosSample,self.nNegSample=triplet_loss_config['nPosSample'],triplet_loss_config['nNegSample']

        # loading training data
        self.database = self.load_database(data_info["database"])
        self.query_data_sets = load_pitts250k_data(data_info["query"], train_conf, self.student_models.models)
        self.query_data_set = ConcatDataset(self.query_data_sets)
        self.train_data_sets = []
        for cities in [CITIES_300x400, CITIES_480x640]:
            train_data_per_res = load_gsv_cities_data(data_info["train"], train_conf, cities, self.student_models.models)
            self.train_data_sets.append(ConcatDataset(train_data_per_res))

        self.similarity_loss = Loss_distill().to(self.device)
        self.vlad_mse_loss = nn.MSELoss().to(self.device)
        self.triplet_loss = nn.TripletMarginLoss(margin=triplet_loss_config["margin"] ** 0.5, p=2, reduction="sum").to(self.device)
    
    def load_database(self, db_info):
        image_folder = db_info["images_path"]
        # Create descriptor file handles and empty descriptor tensors
        g_desc_files = {model: h5py.File(join(image_folder, f"global_descriptor_{model}.h5"), "r") for model in self.teacher_models.models}
        num_images = len(g_desc_files[list(g_desc_files.keys())[0]][basename(image_folder)])
        assert all([num_images == len(file[basename(image_folder)]) for file in g_desc_files.values()]), \
            f"Database global descriptors for {image_folder} have different lengths (first file has length {num_images})."
        descriptors = {model: torch.empty((len(db_info["img_names"]), self.teacher_models.feature_length(model))) for model in self.teacher_models.models}
        locations = torch.empty((len(db_info["img_names"]), 2))
        img_paths = []
        for file_ind, (model, g_desc_file) in enumerate(g_desc_files.items()):
            for i, img_name in enumerate(tqdm(db_info["img_names"], desc=f"Reading database info for {model}")):
                name, pitch, yaw = img_name.replace(".jpg", "").split("_")
                pitch = str((int(pitch.replace("pitch", "")) - 1) * 30).zfill(3)
                yaw = str((int(yaw.replace("yaw", "")) - 1) * 30).zfill(3)
                descriptor = g_desc_file[basename(image_folder)][f"{pitch}+{yaw}+{name}.jpg"]
                if file_ind == 0:
                    name, pitch, yaw = img_name.replace(".jpg", "").split("_")
                    pitch = str((int(pitch.replace("pitch", "")) - 1) * 30).zfill(3)
                    yaw = str((int(yaw.replace("yaw", "")) - 1) * 30).zfill(3)
                    image_path = join(image_folder, pitch, yaw, "raw", name + ".jpg")
                    img_paths.append(image_path)
                    locations[i] = torch.tensor(db_info["groundtruth"][i])
                descriptors[model][i] = torch.tensor(descriptor.__array__())
            g_desc_file.close()

        return {"images_path": img_paths, "descriptors": descriptors, "locations": locations}

    def train_loop(self, train_data_set, scaler, schedulers, pbar, epoch):
        # All training images are used
        batch_sampler = BatchSampler([list(range(len(train_data_set)))], self.batch_size)
        data_loader = DataLoader(train_data_set, batch_sampler=batch_sampler, num_workers=self.train_conf["num_worker"], collate_fn=collate_fn_gsv, pin_memory=True)

        enabled_loss_types = set(loss_type for loss_type, enabled in self.train_conf["loss"].items() if enabled).union(["loss"])
        models_losses = {model: {loss_type: 0 for loss_type in enabled_loss_types} for model in self.student_models.models}
        for images_high, images_low, _, _ in data_loader:
            for optimizer in self.optimizers.values(): optimizer.zero_grad()
            curr_losses = {model: {loss_type: 0 for loss_type in enabled_loss_types} for model in models_losses}
            features_low, vectors_low = self._compute_image_descriptors(images_low, self.student_models, models_losses.keys(), True)
            with torch.no_grad():
                features_high, vectors_high = self._compute_image_descriptors(images_high, self.teacher_models, models_losses.keys(), True)
            for model, losses in curr_losses.items():
                with torch.autocast("cuda", torch.float32):
                    if "distill" in enabled_loss_types:
                        loss_sp = self.similarity_loss(features_low[model], features_high[model]) / 1000
                        losses["distill"] += loss_sp
                        losses["loss"] += loss_sp
                    if "vlad" in enabled_loss_types:
                        loss_vlad = self.vlad_mse_loss(vectors_low[model], vectors_high[model]) * len(images_high) * 100
                        losses["vlad"] += loss_vlad
                        losses["loss"] += loss_vlad
                    if "triplet" in enabled_loss_types:
                        loss_triplet, vector_low = 0, vectors_low[model].view(*images_high[model].shape[:2], -1)
                        for pos_neg_vecs in vector_low:
                            vladQ, vladP, vladN = torch.split(pos_neg_vecs, [1, self.nPosSample, self.nNegSample])
                            vladQ = vladQ.squeeze()
                            for vP in vladP:
                                for vN in vladN:
                                    loss_triplet += self.triplet_loss(vladQ, vP, vN)
                        loss_triplet /= (len(images_high) * self.nPosSample * self.nNegSample * 10)
                        losses["triplet"] += loss_triplet
                        losses["loss"] += loss_triplet
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(self.optimizers[model])
                clip_grad_norm_(self.student_models.model_parameters(model), 1, foreach=True)
                scaler.step(self.optimizers[model])
                schedulers[model].step()
                scaler.update()

            loss_desc = " ".join([f"{model} ep#{self.start_epochs[model] + epoch} loss: {losses['loss']:.4e} (lr: {self.optimizers[model].param_groups[0]['lr']:.4e})"
                                    for model, losses in curr_losses.items()])
            progress_desc = f"Epoch {epoch + 1}/{self.train_conf['nepoch']} " + loss_desc
            pbar.set_description(progress_desc, refresh=False)
            pbar.update(1)

            for model, losses in curr_losses.items():
                for loss_name, loss_value in losses.items():
                    models_losses[model][loss_name] += loss_value
            del images_high, images_low, curr_losses, features_low, vectors_low, features_high, vectors_high
            torch.cuda.empty_cache()
        return models_losses

    def train_student(self, epoch):
        
        self.optimizers = {model: optim.Adam(self.student_models.model_parameters(model), lr=self.lr, weight_decay=self.lr_decay, foreach=True)
                           for model in self.student_models.models}

        schedulers = {model: ExponentialLR(optimizer, gamma=self.gamma) for model, optimizer in self.optimizers.items()}
        self.student_models.set_train(True)

        scaler = GradScaler()

        image_total = sum([len(self.train_data_sets[ds_ind]) for ds_ind in range(len(self.train_data_sets))])
        progress_bar = tqdm(total=image_total // self.batch_size + 1)
        total_losses = self.train_loop(self.train_data_sets[0], scaler, schedulers, progress_bar, epoch)
        for ds_ind in range(1, len(self.train_data_sets)):
            models_losses = self.train_loop(self.train_data_sets[ds_ind], scaler, schedulers, progress_bar, epoch)
            for model, losses in models_losses.items():
                for loss_name, loss_value in losses.items():
                    total_losses[model][loss_name] += loss_value
        progress_bar.close()
        
        for model, losses in total_losses.items():
            for loss_type, loss_value in losses.items():
                self.log_writers[model].add_scalar(f"Train/{'Loss' if loss_type == 'loss' else f'Loss_{loss_type}'}",
                                                   loss_value / image_total, self.start_epochs[model] + epoch)
        

    def vpr_examing(self,query_desc,database_desc, query_loc, database_loc):
        query_desc,database_desc, query_loc, database_loc=query_desc.to(self.device),database_desc.to(self.device), query_loc.float().to(self.device), database_loc.float().to(self.device)
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

    def validation(self, ind):
        print("start vpr examining ...")

        self.student_models.set_train(False)

        random.seed(10)

        # Choosing a fixed number of images from each query dataset is currently unused
        # sample_size=1000
        # len_init=self.query_data_sets[0].__len__()
        # indices=[random.sample(list(range(len_init)),sample_size)]
        # for d in self.query_data_sets[1:]:
        #     len_current=len_init+d.__len__()
        #     indices.append(random.sample(list(range(len_init,len_current)),sample_size))
        #     len_init=len_current
        
        # All query images are evaluated for VPR recall
        query_batch=20
        batch_sampler=BatchSampler([list(range(len(self.query_data_set)))], query_batch)
        data_loader = DataLoader(self.query_data_set, batch_sampler=batch_sampler, num_workers=self.train_conf["num_worker"], collate_fn=collate_fn, pin_memory=True)
        total = len(self.query_data_set) // query_batch + 1

        recall_score = {model: {version: torch.zeros((self.thresholds.shape[0], self.topk_nodes.shape[0])) for version in ["teacher", "student"]} for model in self.teacher_models.models}
        with torch.no_grad():
            for _, images_low, images_low_path, locations in tqdm(data_loader, total=total):
                features_low, vectors_low = self._compute_image_descriptors(images_low, self.student_models, recall_score.keys())
                features_teacher_low, vectors_teacher_low = self._compute_image_descriptors(images_low, self.teacher_models, recall_score.keys())

                for model in vectors_low:
                    locations_model = locations[model][:,0,:]
                    recall_score[model]["student"] += self.vpr_examing(vectors_low[model], self.database["descriptors"][model], locations_model, self.database["locations"])
                    recall_score[model]["teacher"] += self.vpr_examing(vectors_teacher_low[model], self.database["descriptors"][model], locations_model, self.database["locations"])
                del images_low, images_low_path, locations, features_teacher_low, features_low, vectors_low, vectors_teacher_low

            recall_rate = {model: {version: recall_score[model][version] / len(self.query_data_set) for version in ["teacher", "student"]} for model in recall_score}

            for model, rate in recall_rate.items():
                for i, topk in enumerate(self.topk_nodes):
                    self.log_writers[model].add_scalars(f"Recall rate/@{int(topk)}", {version: rate[version][0,i] for version in rate}, self.start_epochs[model] + ind)

            for model in self.student_models.models:
                new_state = {"epoch": self.start_epochs[model] + ind, "best_score": recall_score[model]["student"][0, 0]}
                save_path = join(self.weight_paths[model], "checkpoint.pth")
                self.student_models.save_state(model, save_path, new_state)
                if new_state["best_score"] >= self.best_scores[model]:
                    shutil.copyfile(save_path, join(self.weight_paths[model], "model_best.pth"))
                    self.best_scores[model] = new_state["best_score"]

            del recall_score, recall_rate
    
    def _compute_image_descriptors(self, input_images, global_extractors, models, calc_pos_neg=False):
        # Remove batch dimension from images
        images = {model: (model_input.view(-1, *model_input.shape[2:]) if calc_pos_neg else
                          model_input[:, 0, :, :, :]).to(self.device) for model, model_input in input_images.items()}
        # Compute encoded features and global descriptors for images
        with torch.autocast("cuda", torch.float32):
            outputs = {model: global_extractors(model, images[model]) for model in models}
        encodings, descriptors = [{model: data[i] for model, data in outputs.items()} for i in range(2)]
        return encodings, descriptors


def main(configs):
    root = configs["root"]

    gt_path = join(root, "logs", "pitts250k", "pittsburgh_query_1000_utm.mat")
    query_gt = scipy.io.loadmat(gt_path)["Cq"].T
    id_path = join(root, "logs", "pitts250k", "pittsburgh_queryID_1000.mat")
    query_id = scipy.io.loadmat(id_path)["query_id"][0]
    query_id = [int(i) - 1 for i in query_id]

    val_info = scipy.io.loadmat(join(root, "logs", "pitts250k", "pitts30k_val.mat"))["dbStruct"].item()
    pitts30k_db, pitts30k_q = [basename(f[0].item()) for f in val_info[1]], [basename(f[0].item()) for f in val_info[3]]
    db_gt = val_info[2].T
    
    data_info = {"train": {"base_path": join(root, "logs", "GSV-Cities")},
                 "database": {"images_path": join(root, "logs", "pitts250k", "database"), "groundtruth": db_gt, "img_names": pitts30k_db},
                 "query": {"image_folder": join(root, "logs", "pitts250k", "query"), "utm": query_gt, "id": query_id, "img_names": pitts30k_q}}
    
    vpr = VPR(configs, data_info, "GSV-Cities")

    for epoch in range(0, configs["train"]["nepoch"]):
        vpr.train_student(epoch)
        vpr.validation(epoch)
    for dataset_per_res in vpr.train_data_sets:
        for dataset in dataset_per_res.datasets:
            dataset.close_neighbors_files()
    for log_writer in vpr.log_writers.values(): log_writer.close()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="../configs/trainer_pitts250.yaml")
    parser.add_argument("-l", "--loss-num", type=int, help="The loss combination to use (index [1-7] for the permutations of the 3 losses)")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if args.loss_num is not None:
        loss_combs = [(True, True, True), (True, True, False), (True, False, True), (True, False, False), (False, True, True), (False, True, False), (False, False, True)]
        for enabled, loss_type in zip(loss_combs[args.loss_num - 1], config["train"]["loss"]):
            config["train"]["loss"][loss_type] = enabled
    main(config)
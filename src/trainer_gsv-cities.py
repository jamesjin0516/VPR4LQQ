import subprocess
import numpy as np
import torch.nn as nn
import yaml
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from os.path import join
from os import makedirs

from dataset.Data_Control import collate_fn, load_gsv_cities_data, BatchSampler
from dataset.gsv_cities_data import read_gt, TRAIN_CITIES

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
        self.topk_nodes=torch.tensor(configs['vpr']['topk'])

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

        self.start_epochs = {model: self.student_models.last_epoch(model) + 1 if train_conf["resume"] else 0 for model in self.student_models.models}
        self.best_scores = {model: self.student_models.best_score(model) if train_conf["resume"] else 0 for model in self.student_models.models}
        triplet_loss_config = train_conf["triplet_loss"]
        self.nPosSample,self.nNegSample=triplet_loss_config['nPosSample'],triplet_loss_config['nNegSample']

        # loading training data
        self.train_data_sets = load_gsv_cities_data(data_info, train_conf, TRAIN_CITIES, self.student_models.models)
        self.train_data_set = ConcatDataset(self.train_data_sets)

        self.similarity_loss = Loss_distill().to(self.device)
        self.vlad_mse_loss = nn.MSELoss().to(self.device)
        self.triplet_loss = nn.TripletMarginLoss(margin=triplet_loss_config["margin"] ** 0.5, p=2, reduction="sum").to(self.device)

    def train_student(self,epoch):
        
        self.optimizers = {model: optim.Adam(self.student_models.model_parameters(model), lr=self.lr, weight_decay=self.lr_decay, foreach=True)
                           for model in self.student_models.models}

        schedulers = {model: ExponentialLR(optimizer, gamma=self.gamma) for model, optimizer in self.optimizers.items()}
        self.student_models.set_train(True)
        
        # All training images are used
        batch_sampler = BatchSampler([list(range(len(self.train_data_set)))], self.batch_size)
        data_loader = DataLoader(self.train_data_set, batch_sampler=batch_sampler, num_workers=self.train_conf["num_worker"], collate_fn=collate_fn, pin_memory=True)
        total = len(self.train_data_set) // self.batch_size + 1

        scaler = GradScaler()

        enabled_loss_types = set(loss_type for loss_type, enabled in self.train_conf["loss"].items() if enabled).union(["loss"])
        models_losses = {model: {loss_type: 0 for loss_type in enabled_loss_types} for model in self.student_models.models}

        for images_high, images_low, _, _ in (pbar:=tqdm(data_loader, total=total)):
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
        
        for model, losses in models_losses.items():
            for loss_type, loss_value in losses.items():
                self.log_writers[model].add_scalar(f"Train/{'Loss' if loss_type == 'loss' else f'Loss_{loss_type}'}",
                                                   loss_value / len(self.train_data_set), self.start_epochs[model] + epoch)
        
        for model in self.student_models.models:
                new_state = {"epoch": self.start_epochs[model] + epoch, "best_score": None}
                save_path = join(self.weight_paths[model], f"checkpoint_epoch{new_state['epoch']}.pth")
                self.student_models.save_state(model, save_path, new_state)

    
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
    longlat_coords, image_names = read_gt(join(root, "logs", "GSV-Cities", "Images"))
    data_info = {"images_path": join(root, "logs", "GSV-Cities", "Images"), "groundtruth": dict(zip(image_names, longlat_coords))}
    
    vpr = VPR(configs, data_info, "GSV-Cities")

    for epoch in range(0, configs["train"]["nepoch"]):
        vpr.train_student(epoch)
    for dataset in vpr.train_data_set.datasets:
        dataset.close_descriptor_files()
    for log_writer in vpr.log_writers.values(): log_writer.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="../configs/trainer_pitts250.yaml")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    main(config)
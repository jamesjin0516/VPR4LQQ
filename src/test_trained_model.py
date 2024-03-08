import argparse
from os.path import join
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
from third_party.pytorch_NetVlad.Feature_Extractor import NetVladFeatureExtractor


class VPRTester:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, root, data_conf, data_folders, vpr_conf, train_conf, model_IO):
        # Assemble logs and weights folder name using training data name and loss and data configurations
        train_data = train_conf["data"]["name"]
        train_outdir = join(train_data + ("_distill" if train_conf["loss"]["distill"] else "") + ("_vlad" if train_conf["loss"]["vlad"] else "") \
                            + ("_triplet" if train_conf["loss"]["triplet"] else ""),
                            str(train_conf["data"]["resolution"]) + "_" + str(train_conf["data"]["qp"]) + "_" + str(train_conf["lr"]))

        log_dir = join(root, model_IO["logs_path"], data_conf["name"] + "_test", "res_" + data_conf["test_res"], train_outdir)
        self.tensorboard = SummaryWriter(log_dir=log_dir)
        self.thresholds = torch.tensor(vpr_conf["threshold"])
        self.topk_nodes = torch.tensor(vpr_conf["topk"])
        self.batch_size = train_conf["batch_size"]
        
        # Load the trained model, picking "model_best.pth.tar" as weights via type="pipeline"
        ckpt_path, model_configs = join(root, model_IO["weights_path"], train_outdir), vpr_conf["global_extractor"]["netvlad"]

        self.vpr_model = NetVladFeatureExtractor(ckpt_path, type="pipeline", arch=model_configs['arch'],
            num_clusters=model_configs['num_clusters'], pooling=model_configs['pooling'], vladv2=model_configs['vladv2'],
            nocuda=model_configs['nocuda'])
        self.vpr_model.model.eval()
        self.pretrained_model = NetVladFeatureExtractor(join(root, model_configs["ckpt_path"]), arch=model_configs['arch'],
            num_clusters=model_configs['num_clusters'], pooling=model_configs['pooling'], vladv2=model_configs['vladv2'],
            nocuda=model_configs['nocuda'])
        self.pretrained_model.model.eval()

        self.train_conf = train_conf
        self.database_images_set = self.load_database_images(data_folders)

        self.query_images_set = TestDataset(data_folders["query"], data_conf["test_res"], train_conf['triplet_loss'])

    def load_database_images(self, data_folders):
        image_folder = data_folders["database"]

        global_descriptors = h5py.File(join(image_folder,'global_descriptor.h5'), 'r')['database']
        
        descriptors=torch.empty((len(global_descriptors),self.train_conf["cluster"]["dimension"] * self.train_conf["num_cluster"]))
        locations=torch.empty((len(global_descriptors),2))
        names = []
        for i,(name,d) in enumerate(tqdm(global_descriptors.items(), desc="Reading database info")):
            splitted = name.split('@')
            utm_east_str, utm_north_str = splitted[1], splitted[2]
            image_path=join(image_folder, 'raw', name)
            names.append(image_path)
            descriptors[i]=torch.tensor(d.__array__())
            locations[i]=torch.tensor([float(utm_east_str), float(utm_north_str)], dtype=torch.float32)

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
        print('start vpr examing ...')

        random.seed(10)
        
        # All query images are evaluated for VPR recall
        query_batch = 20
        batch_sampler = BatchSampler([list(range(len(self.query_images_set)))], query_batch)
        data_loader = DataLoader(self.query_images_set, collate_fn=collate_fn, num_workers=self.train_conf['num_worker'], pin_memory=True, batch_sampler=batch_sampler)

        recall_score, r_pretrained_score = torch.zeros((self.thresholds.shape[0],self.topk_nodes.shape[0])), torch.zeros((self.thresholds.shape[0],self.topk_nodes.shape[0]))
        with torch.no_grad():
            for images_high, images_low, locations in tqdm(data_loader, total=len(self.query_images_set) // query_batch + 1):
                # Batches containing identical resolution images are stacked into tensors
                if isinstance(images_low, torch.Tensor):
                    images_low = images_low[:, 0, 0, :, :, :].to(self.device)
                # otherwise batches contain lists, which each have the query image and its positive & negative images
                else:
                    images_low = [batch_item[0].to(self.device) for batch_item in images_low]
                locations = locations[:, 0, :]
                # Compute global descriptors for low resolution images
                with torch.autocast('cuda', torch.float32):
                    if isinstance(images_low, torch.Tensor):
                        features_low = self.vpr_model.model.encoder(images_low)
                        f_pretrained_low = self.pretrained_model.model.encoder(images_low)
                        vectors_low = self.vpr_model.model.pool(features_low).detach().cpu()
                        v_pretrained_low = self.pretrained_model.model.pool(f_pretrained_low).detach().cpu()
                    else:
                        # Given different query image resolutions, compute descriptors sequentially
                        features_low = [self.vpr_model.model.encoder(image_low) for image_low in images_low]
                        f_pretrained_low = [self.pretrained_model.model.encoder(image_low) for image_low in images_low]
                        vectors_low = torch.cat([self.vpr_model.model.pool(feature_low).detach().cpu() for feature_low in features_low])
                        v_pretrained_low = torch.cat([self.pretrained_model.model.pool(feature_low).detach().cpu() for feature_low in f_pretrained_low])

                recall_score += self.vpr_examing(vectors_low, self.database_images_set['descriptors'], locations, self.database_images_set['locations'])
                r_pretrained_score += self.vpr_examing(v_pretrained_low, self.database_images_set['descriptors'], locations, self.database_images_set['locations'])
                del images_low, locations, features_low, vectors_low, f_pretrained_low, v_pretrained_low

            recall_rate, r_pretrained_rate = recall_score / len(self.query_images_set), r_pretrained_score / len(self.query_images_set)

            for i,topk in enumerate(self.topk_nodes):
                self.tensorboard.add_scalars(f'Recall rate/@{int(topk)}', {'trained model': recall_rate[0, i], "pretrained": r_pretrained_rate[0, i]}, iter_num)
            
            del recall_score, recall_rate, r_pretrained_score, r_pretrained_rate
        torch.cuda.empty_cache()

        if not hasattr(self, "valid_data_sets"): return
        print("Start loss validation ...")
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
            Loss_dict = {'loss': 0}
            if self.train_conf['loss']['distill']:
                Loss_dict['distill'] = 0
            if self.train_conf['loss']['vlad']:
                Loss_dict['vlad'] = 0
            if self.train_conf['loss']['triplet']:
                Loss_dict['triplet'] = 0
            for images_high, images_low, _ in tqdm(data_loader, total=len(self.valid_data_set) // self.batch_size + 1):
                images_high, images_low = images_high.to(self.device), images_low.to(self.device)
                B, G, C, HH, WH = images_high.size()
                B, G, C, HL, WL = images_low.size()
                images_high = images_high.view(B*G, C, HH, WH)
                images_low = images_low.view(B*G, C, HL, WL)
                with torch.autocast('cuda', torch.float32):
                    features_low = self.vpr_model.model.encoder(images_low)
                    vectors_low = self.vpr_model.model.pool(features_low)
                    features_high = self.pretrained_model.model.encoder(images_high)
                    vectors_high = self.pretrained_model.model.pool(features_high)

                    Loss_vlad = self.vlad_mse_loss(vectors_low, vectors_high).detach().cpu() * B * 100
                    Loss_sp = self.similarity_loss(features_low, features_high).detach().cpu() / 1000

                    _, D = vectors_low.size()
                    vectors_low = vectors_low.view(B, G, D)
                    Loss_triplet = 0
                    for vector_low in vectors_low:
                        vladQ, vladP, vladN = torch.split(vector_low, [1, self.nPosSample, self.nNegSample])
                        vladQ = vladQ.squeeze()
                        for vP in vladP:
                            for vN in vladN:
                                Loss_triplet += self.triplet_loss(vladQ, vP, vN)

                    Loss_triplet /= (self.nPosSample * self.nNegSample * 10)
                    Loss_triplet = Loss_triplet.detach().cpu()

                    Loss= Loss_sp + Loss_vlad + Loss_triplet

                    Loss_dict['loss'] += Loss
                    if self.train_conf['loss']['distill']:
                        Loss_dict['distill'] += Loss_sp
                    if self.train_conf['loss']['vlad']:
                        Loss_dict['vlad'] += Loss_vlad
                    if self.train_conf['loss']['triplet']:
                        Loss_dict['triplet'] += Loss_triplet

                del images_high,images_low,vectors_low,vectors_high,vector_teacher_low
                torch.cuda.empty_cache()

            self.tensorboard.add_scalar('Valid/Loss', Loss_dict['loss'] / int(sample_size * len(self.valid_data_sets)), iter_num)
            Loss_dict.pop('loss')
            for k, v in Loss_dict.items():
                self.tensorboard.add_scalar(f'Valid/Loss_{k}', v / int(sample_size * len(self.valid_data_sets)), iter_num)


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

    for iter_num in range(1, configs["test_runs"] + 1):
        vpr.validation(iter_num)
    vpr.tensorboard.close()
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../configs/test_trained_model.yaml')
    parser.add_argument("-d", "--data_info", type=str, default="../configs/testing_data.yaml")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    with open(args.data_info, "r") as d_locs_file:
        data_info = yaml.safe_load(d_locs_file)
    main(config, data_info)
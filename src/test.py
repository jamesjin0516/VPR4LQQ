import subprocess
import numpy as np
import torch.nn as nn
import yaml
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from feature.Geomatric_verification import Geometric_verification
from os.path import join,exists
from os import makedirs,listdir

from dataset.Data_Control import data_link,load_test_data,BatchSampler

from loss.loss_distill import Loss_distill
from math import ceil
import h5py
import sys
import faiss
sys.path.append(join(sys.path[0],'..'))
from third_party.pytorch_NetVlad.Feature_Extractor import NetVladFeatureExtractor
from tensorboardX import SummaryWriter
from datetime import datetime
from tqdm.auto import tqdm
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import ConcatDataset
import torch
import torch.backends.cuda as cuda
import torch.backends.cudnn as cudnn
import torch.backends.opt_einsum as opt_einsum
import json
import random
import scipy.io
import shutil

cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True
cudnn.benchmark = True
cuda.matmul.allow_fp16_reduced_precision_reduction = True
cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.set_default_dtype(torch.float32)
opt_einsum.enabled = True
cudnn.enabled = True

def split_integer(x,n):
    quotient,remainder=divmod(x,n)
    slices=[quotient for _ in range(n)]
    for i in range(remainder):
        slices[i]+=1
    return slices

class VPR():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self,configs, data_path,data_name,model_path,models):
        self.config=configs
        self.batch_size=configs['train']['batch_size']
        self.lr=configs['train']['lr']
        self.lr_decay=configs['train']['lr_decay']
        self.gamma=configs['train']['exponential_gamma']

        self.thresholds = torch.tensor(configs['vpr']['threshold'])

        self.topk_nodes=torch.tensor(configs['vpr']['topk'])
        self.model_path=model_path
        self.models=models

        logs_dir=join(configs['root'],'logs','tensorboard_logs',data_name)
        if not exists(logs_dir):
            makedirs(logs_dir)
            
        log_dir=data_name
        self.save_path=join(configs['root'],'parameters/RA_VPR',data_name)
        if self.config['train']['loss']['distill']:
            log_dir+='_distill'
            self.save_path+='_distill'
        if self.config['train']['loss']['vlad']:
            log_dir+='_vlad'
            self.save_path+='_vlad'
        if self.config['train']['loss']['triplet']:
            log_dir+='_triplet'
            self.save_path+='_triplet'
        if not exists(self.save_path):
            makedirs(self.save_path)

        self.save_path=join(self.save_path,str(self.config['train']['data']['resolution'])+'_'+str(self.config['train']['data']['qp'])+'_'+str(self.lr))
        if not exists(self.save_path):
            makedirs(self.save_path)
        log_dir=join(log_dir,str(self.config['train']['data']['resolution'])+'_'+str(self.config['train']['data']['qp'])+'_'+str(self.lr))

        content=configs['vpr']['global_extractor']['netvlad']
        self.content=content
        ckpt_path=join(configs['root'], content['ckpt_path'], 'checkpoints')

        self.teacher_model=NetVladFeatureExtractor(ckpt_path,'baseline', arch=content['arch'],
         num_clusters=content['num_clusters'],
         pooling=content['pooling'], vladv2=content['vladv2'], nocuda=content['nocuda'])
        self.teacher_model.model.to(self.device).eval()
        self.teacher_model.model.requires_grad_(False)
        self.teacher_model.model.to(torch.float32)
        self.teacher_model.model = torch.compile(self.teacher_model.model, fullgraph=True, dynamic=False, mode='max-autotune', backend='onnxrt')

        ### loading database / query data
        self.databases=self.load_database(data_path['database'])

        for k,v in data_path['query'].items():
            image_folder=v['image_folder']
            self.query_data_sets=load_test_data(image_folder,**configs['train'])
        self.geometric_verification = Geometric_verification()

    #@pysnooper.snoop('train-VID.log', depth=3)
    def load_database(self,databases_config):
        databases={}
        for k,v in databases_config.items():
            image_folder=v['image_folder']
            global_descriptors = h5py.File(join(image_folder,'global_descriptor.h5'), 'r')[k]
            gt_path=v['utm_file']
            scale=v['scale']
            gt=self.get_gt(gt_path,scale)
            descriptors=torch.empty((len(global_descriptors),self.config['train']['cluster']['dimension']*self.config['train']['num_cluster']))
            locations=torch.empty((len(global_descriptors),2))
            names=[]
            for i,(name,d) in enumerate(global_descriptors.items()):
                pitch,yaw,image_name=name.split('+')
                resolution=sorted(listdir(join(image_folder,pitch,yaw,'images')))[-1]
                image_path=join(image_folder,pitch,yaw,'images',resolution,'raw',image_name)
                names.append(image_path)
                descriptors[i]=torch.tensor(d.__array__())
                locations[i]=torch.tensor(gt[name.replace('.png','').split('+')[-1]][:2])
            database_name=k.split('_')[0]
            if database_name in databases:
                databases[database_name]['images_path']=databases[database_name]['images_path']+names
                databases[database_name]['descriptors']=torch.cat(databases[database_name]['descriptors'],descriptors)
                databases[database_name]['locations']=torch.cat(databases[database_name]['locations'],locations)
            else:
                databases[database_name]={'images_path':names,'descriptors':descriptors,'locations':locations}
        return databases
    
    def get_gt(self,aligner_path,scale):
        with open(aligner_path, "r") as f:
            keyframes = json.load(f)['keyframes']
        key= {}
        for id, point in keyframes.items():
            t_mp=point['trans']
            rot=point['rot']
            key[str(int(id)-1).zfill(5)] = [t_mp[0]*scale, t_mp[1]*scale, rot]
        return key
    
    def vpr_examing(self,query_desc,database_desc, query_loc, database_loc,failed_file,model):
        query_desc,database_desc, query_loc, database_loc=query_desc.to(self.device),database_desc.to(self.device), query_loc.float().to(self.device), database_loc.float().to(self.device)
        sim = torch.einsum('id,jd->ij', query_desc, database_desc)
        topk_ = torch.topk(sim, self.topk_nodes[-1], dim=1)
        topk=topk_.indices
        success_num = torch.zeros((self.thresholds.shape[0],self.topk_nodes.shape[0]))
        for i, index in enumerate(topk):
            test_image_path = self.images_low_path[i]
            train_image_paths = [self.images_high_path[indd] for indd in index]
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
                    ind=torch.stack(list(ind),dim=0).squeeze(0).detach().cpu()
                    if len(ind) > 0:
                        train_image_paths = [self.images_high_path[indd] for indd in index_match]
                        for index_, index_match_,train_image_path in zip(ind,index_match, train_image_paths):
                            if self.geometric_verification.geometric_verification(train_image_path, test_image_path):
                                maskA = torch.zeros((self.thresholds.shape[0], self.topk_nodes.shape[0]), dtype=torch.bool)
                                maskB = torch.zeros((self.thresholds.shape[0], self.topk_nodes.shape[0]), dtype=torch.bool)
                                maskA[self.thresholds >= thre, :] = True
                                maskB[:, self.topk_nodes > index_] = True
                                mask = (maskA & maskB)
                                success_num[mask] += 1
                                matched = True
                                break
            if not matched:
                # failed_file.write(model+'_'+test_image_path+'\n')
                pass


            del qloc,dloc,distance
        del sim,topk,query_desc,database_desc, query_loc, database_loc
        torch.cuda.empty_cache()
        return success_num

    def validation(self):
        print('start vpr examing ...')
        model_names=['Baseline','Triplet','MSE+ICKD']
        sample_size=200
        f=open('/home/unav/Desktop/Resolution_Agnostic_VPR/log/out_unav.txt','a')
        failed_baseline_file=open('/home/unav/Desktop/Resolution_Agnostic_VPR/log/fail_baseline.txt','a')
        failed_distill_file=open('/home/unav/Desktop/Resolution_Agnostic_VPR/log/fail_pipeline.txt','a')
        for resolution,v in self.query_data_sets.items():
            if resolution=='810p':
                continue
            print(resolution)
            f.write(f'Resolution {resolution}:\n')
            for qp, query_datasets in v.items():
                print(qp)
                f.write(f'Quantization parameter {qp}:\n')
                for ii,m in enumerate(models):
                    model_name=model_names[ii]
                    random.seed(10)
                    ckpt_student_path=join(self.model_path,m)
                    self.student_model=NetVladFeatureExtractor(ckpt_student_path,'pipeline', arch=self.content['arch'],
                    num_clusters=self.content['num_clusters'],
                    pooling=self.content['pooling'], vladv2=self.content['vladv2'], nocuda=self.content['nocuda'])
                    self.student_model.model.to(self.device).eval()
                    self.student_model.model.requires_grad_(False)
                    self.student_model.model.to(torch.float32)
                    self.student_model.model = torch.compile(self.student_model.model, fullgraph=True, dynamic=False, mode='max-autotune', backend='onnxrt')
                    len_init=query_datasets.__len__()
                    indices=[random.sample(list(range(len_init)),sample_size)]
                    query_batch=20
                    batch_sampler=BatchSampler(indices,query_batch)
                    data_loader=DataLoader(query_datasets, num_workers=self.config['train']['num_worker'],pin_memory=True,batch_sampler=batch_sampler)

                    evaluation_distill=torch.zeros((self.thresholds.shape[0],self.topk_nodes.shape[0]))
                    evaluation_base=torch.zeros((self.thresholds.shape[0],self.topk_nodes.shape[0]))
                    query_num=0
                    with torch.no_grad():
                        for i,[_,images_low,images_low_path,locations] in enumerate(data_loader):
                            self.images_low_path=images_low_path[0]
                            data_name=self.images_low_path[0].split('/')[-7].split('_')[0]
                            database=self.databases[data_name]
                            self.images_high_path=database['images_path']
                            images_low=images_low[:,0,:,:,:]
                            images_low=images_low.to(self.device)
                            locations=locations[:,0,:]
                            with torch.autocast('cuda', torch.float32):
                                features_low=self.student_model.model.encoder(images_low)
                                vectors_low=self.student_model.model.pool(features_low).detach().cpu()

                                features_teacher_low=self.teacher_model.model.encoder(images_low)
                                vectors_teacher_low=self.teacher_model.model.pool(features_teacher_low).detach().cpu()

                            query_num+=vectors_low.size(0)
                            
                            evaluation_distill+=self.vpr_examing(vectors_low,database['descriptors'],locations,database['locations'],failed_distill_file,model_name)
                            # if ii==0:
                            evaluation_base+=self.vpr_examing(vectors_teacher_low,database['descriptors'],locations,database['locations'],failed_baseline_file,'Baseline')
                            del images_low,images_low_path,locations,self.images_low_path,self.images_high_path,features_teacher_low,features_low,vectors_low,vectors_teacher_low

                        evaluation_distill=evaluation_distill/sample_size
                        evaluation_base=evaluation_base/sample_size
                        f.write(f'Baseline:\n Recall rate/@{int(self.topk_nodes[0])}/threshold {int(self.thresholds[0])}m,Recall rate/@{int(self.topk_nodes[0])}/threshold {int(self.thresholds[-1])}m,Recall rate/@{int(self.topk_nodes[0])}/@{int(self.topk_nodes[-1])}/threshold {int(self.thresholds[0])}m,Recall rate/@{int(self.topk_nodes[-1])}/threshold {int(self.thresholds[-1])}m:\n[{evaluation_base[0,0]},{evaluation_base[-1,0]},{evaluation_base[0,-1]},{evaluation_base[-1,-1]}]\n')
                        f.write(f'{m}:\n Recall rate/@{int(self.topk_nodes[0])}/threshold {int(self.thresholds[0])}m,Recall rate/@{int(self.topk_nodes[0])}/threshold {int(self.thresholds[-1])}m,Recall rate/@{int(self.topk_nodes[-1])}/threshold {int(self.thresholds[0])}m,Recall rate/@{int(self.topk_nodes[-1])}/threshold {int(self.thresholds[-1])}m:\n[{evaluation_distill[0,0]},{evaluation_distill[-1,0]},{evaluation_distill[0,-1]},{evaluation_distill[-1,-1]}]\n\n')
                        print(f'Baseline:\n Recall rate/@{int(self.topk_nodes[0])}/threshold {int(self.thresholds[0])}m,Recall rate/@{int(self.topk_nodes[-1])}/threshold {int(self.thresholds[0])}m,Recall rate/@{int(self.topk_nodes[0])}/threshold {int(self.thresholds[-1])}m,Recall rate/@{int(self.topk_nodes[-1])}/threshold {int(self.thresholds[-1])}m:\n[{evaluation_base[0,0]},{evaluation_base[0,-1]},{evaluation_base[-1,0]},{evaluation_base[-1,-1]}]')
                        print(f'{m}:\n Recall rate/@{int(self.topk_nodes[0])}/threshold {int(self.thresholds[0])}m,Recall rate/@{int(self.topk_nodes[-1])}/threshold {int(self.thresholds[0])}m,Recall rate/@{int(self.topk_nodes[0])}/threshold {int(self.thresholds[-1])}m,Recall rate/@{int(self.topk_nodes[-1])}/threshold {int(self.thresholds[-1])}m:\n[{evaluation_distill[0,0]},{evaluation_distill[0,-1]},{evaluation_distill[-1,0]},{evaluation_distill[-1,-1]}]')
                    del self.student_model
                    torch.cuda.empty_cache()
        f.close()
        failed_baseline_file.close()
        failed_distill_file.close()


def main(configs,model_path,models):
    root = configs['root']
    with open(join(root,'configs','unav_train_data.yaml'), 'r') as f:
        data_split_config = yaml.safe_load(f)
    with open(join(root,'configs','measurement_scale.yaml'), 'r') as f:
        scales = yaml.safe_load(f)

    database_folder_list=data_split_config['database']
    query_folder_list=data_split_config['query']

    data_path={}

    data_path['database']=data_link(root,data_split_config['name'],scales,database_folder_list)
    data_path['query']=data_link(root,data_split_config['name'],scales,query_folder_list)

    vpr=VPR(configs,data_path,data_split_config['name'],model_path,models)
    vpr.validation()
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/home/unav/Desktop/Resolution_Agnostic_VPR/configs/trainer.yaml')
    parser.add_argument('-p', '--model_path', type=str, default=None)
    # parser.add_argument('-m','--models', action='append', help='<Required> Set flag', required=True)
    parser.add_argument('-m','--models', nargs='+', default=[])
    # parser.add_argument('-m','--models', nargs='+', help='<Required> Set flag', required=True)
    args = parser.parse_args()
    model_path=args.model_path
    models=args.models[0].split(',')
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    main(config,model_path,models)
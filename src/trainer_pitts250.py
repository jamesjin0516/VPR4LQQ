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

from dataset.Data_Control import load_pitts250k_data,BatchSampler

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
    def __init__(self,configs, data_path,data_name):
        self.config=configs
        self.batch_size=configs['train']['batch_size']
        self.lr=configs['train']['lr']
        self.lr_decay=configs['train']['lr_decay']
        self.gamma=configs['train']['exponential_gamma']

        self.thresholds = torch.tensor(configs['vpr']['threshold'])

        self.topk_nodes=torch.tensor(configs['vpr']['topk'])

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

        self.writer = SummaryWriter(log_dir=join(logs_dir, log_dir))

        content=configs['vpr']['global_extractor']['netvlad']
        
        ckpt_path=join(configs['root'], content['ckpt_path'], 'checkpoints')

        if configs['train']['resume']:
            ckpt_student_path=self.save_path
        else:
            ckpt_student_path=ckpt_path

        self.teacher_model=NetVladFeatureExtractor(ckpt_path, arch=content['arch'],
         num_clusters=content['num_clusters'],
         pooling=content['pooling'], vladv2=content['vladv2'], nocuda=content['nocuda'])
        self.teacher_model.model.to(self.device).eval()
        self.teacher_model.model.requires_grad_(False)
        self.teacher_model.model.to(torch.float32)
        self.teacher_model.model = torch.compile(self.teacher_model.model, fullgraph=True, dynamic=False, mode='max-autotune', backend='onnxrt')

        self.student_model=NetVladFeatureExtractor(ckpt_student_path, arch=content['arch'],
         num_clusters=content['num_clusters'],
         pooling=content['pooling'], vladv2=content['vladv2'], nocuda=content['nocuda'])
        if configs['train']['resume']:
            self.start_epoch=self.student_model.checkpoint['epoch']+1
            self.best_score=self.student_model.checkpoint['best_score']
        else:
            self.start_epoch=0
            self.best_score=0
        # with h5py.File(self.initcache, mode='r') as h5: 
        #     clsts = h5.get("centroids")[...]
        #     traindescs = h5.get("descriptors")[...]
        #     self.student_model.model.pool.init_params(clsts, traindescs)
        #     del clsts, traindescs
        self.student_model.model.to(self.device)
        self.student_model.model = torch.compile(self.student_model.model, fullgraph=True, dynamic=False, mode='max-autotune', backend='aot_ts_nvfuser')
        
        triplet_loss_config=configs['train']['triplet_loss']

        self.nPosSample,self.nNegSample=triplet_loss_config['nPosSample'],triplet_loss_config['nNegSample']

        ### loading database / query data
        self.databases=self.load_database(data_path['database'])
        self.query_data_sets=load_pitts250k_data(data_path['query'],configs['train'])
        self.train_data_sets=load_pitts250k_data(data_path['train'],configs['train'])
        self.valid_data_sets=load_pitts250k_data(data_path['valid'],configs['train'])

        self.query_data_set=ConcatDataset(self.query_data_sets)
        self.train_data_set=ConcatDataset(self.train_data_sets)
        self.valid_data_set=ConcatDataset(self.valid_data_sets)

        self.initcache = join(configs['root'],'logs',data_name,'VID' + '_' + str(configs['train']['num_cluster']) + '_'+str(configs['train']['data']['resolution']) +'_'+str(configs['train']['data']['qp'])+'.hdf5')


        self.geometric_verification = Geometric_verification()

        self.similarity_loss=Loss_distill().to(self.device)
        self.vlad_mse_loss=nn.MSELoss().to(self.device)
        self.triplet_loss= nn.TripletMarginLoss(margin=configs['train']['triplet_loss']['margin']**0.5, 
                p=2, reduction='sum').to(self.device)
        
        self.sigmoid = nn.Sigmoid()

    #@pysnooper.snoop('train-VID.log', depth=3)
    def load_database(self,v):
        databases={}

        image_folder=v['image_folder']
        global_descriptors = h5py.File(join(image_folder,'global_descriptor.h5'), 'r')['database']

        gt=v['utm']
        gt_id=v['id']
        descriptors=torch.empty((len(global_descriptors),self.config['train']['cluster']['dimension']*self.config['train']['num_cluster']))
        locations=torch.empty((len(global_descriptors),2))
        names=[]
        for i,(name,d) in enumerate(global_descriptors.items()):
            pitch,yaw,image_name=name.split('+')
            image_path=join(image_folder,pitch,yaw,'raw',image_name)
            names.append(image_path)
            descriptors[i]=torch.tensor(d.__array__())
            locations[i]=torch.tensor(gt[gt_id.index(int(image_name.replace('.jpg','')))])

        databases['database']={'images_path':names,'descriptors':descriptors,'locations':locations}
        return databases
            
    def train_student(self,epoch):
        
        self.optimizer = optim.Adam(self.student_model.model.parameters(), lr=self.lr, weight_decay=self.lr_decay, foreach=True)

        scheduler = ExponentialLR(self.optimizer, gamma=self.gamma)
        self.student_model.model.train()
        
        sample_size=40000//len(self.train_data_sets)
        len_init=self.train_data_sets[0].__len__()
        indices=[random.sample(list(range(len_init)),sample_size)]
        for d in self.train_data_sets[1:]:
            len_current=len_init+d.__len__()
            indices.append(random.sample(list(range(len_init,len_current)),sample_size))
            len_init=len_current

        # len_init=self.train_data_sets[0].__len__()
        # indices=[list(range(len_init))]
        # for d in self.train_data_sets[1:]:
        #     len_current=len_init+d.__len__()
        #     indices.append(list(range(len_init,len_current)))
        #     len_init=len_current
        batch_sampler=BatchSampler(indices,self.batch_size)
        data_loader=DataLoader(self.train_data_set, num_workers=self.config['train']['num_worker'],pin_memory=True,batch_sampler=batch_sampler)
        total=sample_size*len(self.train_data_sets)//self.batch_size+1
        
        #total = self.train_data_set.__len__() // self.batch_size + 1

        scaler = GradScaler()

        Loss_dict={'loss':0}
        if self.config['train']['loss']['distill']:
            Loss_dict['distill']=0
        if self.config['train']['loss']['vlad']:
            Loss_dict['vlad']=0
        if self.config['train']['loss']['triplet']:
            Loss_dict['triplet']=0

        for i, [images_high,images_low,_,_] in enumerate(pbar:=tqdm(data_loader, total=total)):
                self.optimizer.zero_grad()
                images_high,images_low=images_high.to(self.device),images_low.to(self.device)
                B,G,C,HH,WH=images_high.size()
                B,G,C,HL,WL=images_low.size()
                images_high=images_high.view(B*G,C,HH,WH)
                images_low=images_low.view(B*G,C,HL,WL)
                with torch.autocast('cuda', torch.float32):
                    features_low=self.student_model.model.encoder(images_low)
                    vectors_low=self.student_model.model.pool(features_low)
                    with torch.no_grad():
                        features_high=self.teacher_model.model.encoder(images_high)
                        vectors_high=self.teacher_model.model.pool(features_high)
                    
                    Loss_vlad=self.vlad_mse_loss(vectors_low,vectors_high)*B*100
                    Loss_sp=self.similarity_loss(features_low, features_high)/1000
                    _,D=vectors_low.size()
                    vectors_low=vectors_low.view(B,G,D)

                    Loss_triplet=0
                    for vector_low in vectors_low:
                        vladQ, vladP, vladN=torch.split(vector_low,[1,self.nPosSample,self.nNegSample])
                        vladQ=vladQ.squeeze()
                        for vP in vladP:
                            for vN in vladN:
                                Loss_triplet+=self.triplet_loss(vladQ,vP,vN)

                    Loss_triplet/=(B*self.nPosSample*self.nNegSample*10)

                    bar=''
                    Loss=0
                    if self.config['train']['loss']['distill']:
                        Loss_dict['distill']+=Loss_sp
                        Loss+=Loss_sp
                        bar+=f'Loss distill {Loss_sp.item():.4e} '
                    if self.config['train']['loss']['vlad']:
                        Loss_dict['vlad']+=Loss_vlad
                        Loss+=Loss_vlad
                        bar+=f'Loss vlad {Loss_vlad.item():.4e} '
                    if self.config['train']['loss']['triplet']:
                        Loss_dict['triplet']+=Loss_triplet
                        Loss+=Loss_triplet
                        bar+=f'Loss triplet {Loss_triplet.item():.4e} '
                    nepoch=self.config['train']['nepoch']
                    bar=f'Epoch {epoch+1}/{nepoch} / Step {i} Loss {Loss.item():.4e} '+bar
                    Loss_dict['loss']+=Loss
                
                scaler.scale(Loss).backward()
                scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.student_model.model.parameters(), 1, foreach=True)
                scaler.step(self.optimizer)
                scheduler.step()
                scaler.update()
                lr= self.optimizer.param_groups[0]['lr']
                bar+=f' Lr {lr:.4e}'
                pbar.set_description(bar, refresh=False)
                pbar.update(1)
                del images_high,images_low,Loss, features_low, vectors_low, features_high, vectors_high
                torch.cuda.empty_cache()
                
        self.writer.add_scalar('Train/Loss', Loss_dict['loss']/self.train_data_set.__len__(), epoch)
        Loss_dict.pop('loss')
        for k,v in Loss_dict.items():
            self.writer.add_scalar(f'Train/Loss_{k}', v/self.train_data_set.__len__(),epoch)

    def vpr_examing(self,query_desc,database_desc, query_loc, database_loc):
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

    def validation(self,ind):
        print('start vpr examing ...')

        self.student_model.model.eval()

        random.seed(10)

        # Choosing a fixed number of images from each query dataset is currently unused
        sample_size=1000
        len_init=self.query_data_sets[0].__len__()
        indices=[random.sample(list(range(len_init)),sample_size)]
        for d in self.query_data_sets[1:]:
            len_current=len_init+d.__len__()
            indices.append(random.sample(list(range(len_init,len_current)),sample_size))
            len_init=len_current
        
        # All query images are evaluated for VPR recall
        query_batch=20
        batch_sampler=BatchSampler([list(range(len(self.query_data_set)))], query_batch)
        data_loader=DataLoader(self.query_data_set, num_workers=self.config['train']['num_worker'],pin_memory=True,batch_sampler=batch_sampler)

        evaluation_distill=torch.zeros((self.thresholds.shape[0],self.topk_nodes.shape[0]))
        evaluation_base=torch.zeros((self.thresholds.shape[0],self.topk_nodes.shape[0]))
        query_num=0
        with torch.no_grad():
            for i,[_,images_low,images_low_path,locations] in enumerate(data_loader):
                print(f'Examing [{i+1}]/[{int(len(self.query_data_set)/query_batch)}]...')
                self.images_low_path=images_low_path[0]
                database=self.databases['database']
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
                evaluation_distill+=self.vpr_examing(vectors_low,database['descriptors'],locations,database['locations'])
                evaluation_base+=self.vpr_examing(vectors_teacher_low,database['descriptors'],locations,database['locations'])
                del images_low,images_low_path,locations,self.images_low_path,self.images_high_path,features_teacher_low,features_low,vectors_low,vectors_teacher_low

            evaluation_distill=evaluation_distill/len(self.query_data_set)
            evaluation_base=evaluation_base/len(self.query_data_set)

            for i,topk in enumerate(self.topk_nodes):
                self.writer.add_scalars(f'Recall rate/@{int(topk)}', {'Baseline': evaluation_base[0,i], 'Distallation': evaluation_distill[0,i]}, ind)

            model_state_dict = self.student_model.model.state_dict()
            state={'epoch': ind,
                            'state_dict': model_state_dict,
                            'best_score': evaluation_distill[0,0],
                        }
            save_path=join(self.save_path,'checkpoint.pth.tar')
            torch.save(state,save_path)
            if evaluation_distill[0,0]>=self.best_score:
                shutil.copyfile(save_path, join(self.save_path, 'model_best.pth.tar'))
                self.best_score=evaluation_distill[0,0]

            del evaluation_base,evaluation_distill

        print('start loss validation ...')
        sample_size=15
        len_init=self.valid_data_sets[0].__len__()
        indices=[random.sample(list(range(len_init)),sample_size)]
        for d in self.valid_data_sets[1:]:
            len_current=len_init+d.__len__()
            indices.append(random.sample(list(range(len_init,len_current)),sample_size))
            len_init=len_current
        batch_sampler=BatchSampler(indices,self.batch_size)
        data_loader=DataLoader(self.valid_data_set, num_workers=self.config['train']['num_worker'],pin_memory=True,batch_sampler=batch_sampler)

        with torch.no_grad():
            Loss_dict={'loss':0}
            if self.config['train']['loss']['distill']:
                Loss_dict['distill']=0
            if self.config['train']['loss']['vlad']:
                Loss_dict['vlad']=0
            if self.config['train']['loss']['triplet']:
                Loss_dict['triplet']=0
            Loss_dict={'loss':0,'distill':0,'vlad':0,'triplet':0}
            for i,[images_high,images_low,_,_] in enumerate(data_loader):
                print(f'Evaluating [{i}]/[{int(sample_size*len(self.valid_data_sets)/self.batch_size)}]...')
                images_high,images_low=images_high.to(self.device),images_low.to(self.device)
                B,G,C,HH,WH=images_high.size()
                B,G,C,HL,WL=images_low.size()
                images_high=images_high.view(B*G,C,HH,WH)
                images_low=images_low.view(B*G,C,HL,WL)
                with torch.autocast('cuda', torch.float32):
                    features_low=self.student_model.model.encoder(images_low)
                    vectors_low=self.student_model.model.pool(features_low)
                    features_high=self.teacher_model.model.encoder(images_high)
                    vectors_high=self.teacher_model.model.pool(features_high)

                    feature_teacher_low=self.teacher_model.model.encoder(images_low)
                    vector_teacher_low=self.teacher_model.model.pool(feature_teacher_low)

                    Loss_vlad=self.vlad_mse_loss(vectors_low,vectors_high).detach().cpu()*B*100
                    Loss_sp=self.similarity_loss(features_low, features_high).detach().cpu()/1000

                    _,D=vectors_low.size()
                    vectors_low=vectors_low.view(B,G,D)
                    Loss_triplet=0
                    for vector_low in vectors_low:
                        vladQ, vladP, vladN=torch.split(vector_low,[1,self.nPosSample,self.nNegSample])
                        vladQ=vladQ.squeeze()
                        for vP in vladP:
                            for vN in vladN:
                                Loss_triplet+=self.triplet_loss(vladQ,vP,vN)

                    Loss_triplet/=(self.nPosSample*self.nNegSample*10)
                    Loss_triplet=Loss_triplet.detach().cpu()

                    Loss=Loss_sp+Loss_vlad+Loss_triplet

                    Loss_dict['loss']+=Loss
                    if self.config['train']['loss']['distill']:
                        Loss_dict['distill']+=Loss_sp
                    if self.config['train']['loss']['vlad']:
                        Loss_dict['vlad']+=Loss_vlad
                    if self.config['train']['loss']['triplet']:
                        Loss_dict['triplet']+=Loss_triplet

                del images_high,images_low,vectors_low,vectors_high,vector_teacher_low
                torch.cuda.empty_cache()
            

            self.writer.add_scalar('Valid/Loss', Loss_dict['loss']/int(sample_size*len(self.valid_data_sets)), ind)
            Loss_dict.pop('loss')
            for k,v in Loss_dict.items():
                self.writer.add_scalar(f'Valid/Loss_{k}', v/int(sample_size*len(self.valid_data_sets)),ind)

        torch.cuda.empty_cache()
      
    def get_cluster(self):
        cluster_config=self.config['train']['cluster']
        descriptor_num=cluster_config['descriptor_num']
        descriptor_per_image=cluster_config['descriptor_per_image']
        image_num=ceil(descriptor_num/descriptor_per_image)
        image_nums=split_integer(image_num,len(self.train_data_sets))

        len_init=self.train_data_sets[0].__len__()
        indices=[random.sample(list(range(len_init)),image_nums[0])]
        for d,num in zip(self.train_data_sets[1:],image_nums[1:]):
            len_current=len_init+d.__len__()
            indices.append(random.sample(list(range(len_init,len_current)),num))
            len_init=len_current
        batch_sampler=BatchSampler(indices,self.batch_size)
        data_loader=DataLoader(self.train_data_set, num_workers=self.config['train']['num_worker'],pin_memory=True,batch_sampler=batch_sampler)

        print('====> Extracting clusters')
        
        with h5py.File(self.initcache, mode='w') as h5:
            with torch.no_grad():
                self.teacher_model.model.eval()
                print('====> Extracting Descriptors')
                dbFeat = h5.create_dataset("descriptors", 
                        [descriptor_num, cluster_config['dimension']], 
                        dtype=np.float32)
                for iteration, [_,images_low,_,_] in enumerate(data_loader, 1):
                    images_low=images_low[:,0,:,:,:].float().to(self.device)
                    image_low_descriptors= self.teacher_model.model.encoder(images_low).view(images_low.size(0), cluster_config['dimension'], -1).permute(0, 2, 1)
                    batchix = (iteration-1)*self.config['train']['batch_size']*descriptor_per_image
                    for ix in range(image_low_descriptors.size(0)):
                        # sample different location for each image in batch
                        sample = np.random.choice(image_low_descriptors.size(1), descriptor_per_image, replace=False)
                        startix = batchix + ix*descriptor_per_image
                        dbFeat[startix:startix+descriptor_per_image, :] = image_low_descriptors[ix, sample, :].detach().cpu().numpy()

                    print("==> Batch ({}/{})".format(iteration, 
                        ceil(image_num/self.batch_size)), flush=True)
                    del image_low_descriptors, images_low
            print('====> Clustering..')
            niter = 100
            kmeans = faiss.Kmeans(cluster_config['dimension'], self.config['train']['num_cluster'], niter=niter, verbose=False)
            kmeans.train(dbFeat[...])

            print('====> Storing centroids', kmeans.centroids.shape)
            h5.create_dataset('centroids', data=kmeans.centroids)
            print('====> Done!')

def extract_frames(config,image_path,video_path,PITCH,YAW):
    if not exists(image_path):
        makedirs(image_path)
        images=join(image_path,f'{PITCH}_{YAW}_%05d.png')
        cmd = [
        'ffmpeg',
        '-i', video_path,
        '-r', str(config['rate']),
        images]
        subprocess.call(cmd)

def main(configs):
    root = configs['root']
    with open(join(root,'configs','pitts250_train_data.yaml'), 'r') as f:
        data_split_config = yaml.safe_load(f)

    name=data_split_config['name']
    train_folder=data_split_config['train'][0]
    valid_folder=data_split_config['valid'][0]
    database_folder=data_split_config['database'][0]
    query_folder=data_split_config['query'][0]

    gt_path=join(root,'logs',name,'pittsburgh_database_10586_utm.mat')
    database_gt = scipy.io.loadmat(gt_path)['Cdb'].T
    database_id=[i for i,_ in enumerate(database_gt)]

    gt_path=join(root,'logs',name,'pittsburgh_query_1000_utm.mat')
    query_gt = scipy.io.loadmat(gt_path)['Cq'].T
    id_path=join(root,'logs',name,'pittsburgh_queryID_1000.mat')
    query_id = scipy.io.loadmat(id_path)['query_id'][0]
    query_id=[int(i)-1 for i in query_id]

    data={}
    data['train']={'image_folder':join(root,'logs',name,train_folder),'utm':database_gt,'id':database_id}
    data['valid']={'image_folder':join(root,'logs',name,valid_folder),'utm':database_gt,'id':database_id}
    data['database']={'image_folder':join(root,'logs',name,database_folder),'utm':database_gt,'id':database_id}
    data['query']={'image_folder':join(root,'logs',name,query_folder),'utm':query_gt,'id':query_id}

    vpr=VPR(configs,data,name)
    if not exists(vpr.initcache):
        vpr.get_cluster()

    for epoch in range(vpr.start_epoch,vpr.start_epoch+configs['train']['nepoch']):
        vpr.train_student(epoch)
        vpr.validation(epoch)
    vpr.writer.close()
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/home/unav/Desktop/Resolution_Agnostic_VPR/configs/trainer.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    main(config)
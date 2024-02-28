import argparse
from os.path import join
import random
from torch.utils.data import DataLoader
import yaml

from dataset.Data_Control import BatchSampler
from dataset.test_dataset import TestDataset, read_coordinates
from third_party.pytorch_NetVlad.Feature_Extractor import NetVladFeatureExtractor


class VPRTester:

    def __init__(self, root, data_folders, model_configs, training_settings):
        
        ckpt_path = ...
        self.vpr_model = NetVladFeatureExtractor(ckpt_path, type="pipeline", arch=model_configs['arch'],
         num_clusters=model_configs['num_clusters'], pooling=model_configs['pooling'], vladv2=model_configs['vladv2'],
         nocuda=model_configs['nocuda'])
        self.vpr_model.model.eval()

        self.database_images_set = self.load_database_images(data_folders)
        assert set("images_path", "descriptors", "locations") == set(self.database_images_set.keys()), f"database keys are incorrect: {self.database.keys()}"

        self.query_images_set = TestDataset(data_folders["query"], training_settings["resolution"])

        # still have to set up tensorboard writer, using training_settings (loss information, training dataset) and root

    def load_database_images(self, data_folders):
        read_coordinates(...)
        pass
    
    def vpr_examing(self, query_desc, database_desc, query_loc, database_loc):
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
        query_batch=20
        batch_sampler=BatchSampler([list(range(len(self.query_images_set)))], query_batch)
        data_loader=DataLoader(self.query_images_set, num_workers=self.config['train']['num_worker'],pin_memory=True,batch_sampler=batch_sampler)

        recall_score = torch.zeros((self.thresholds.shape[0],self.topk_nodes.shape[0]))
        with torch.no_grad():
            for i, [images_high, images_low, locations] in enumerate(data_loader):
                print(f'Examing [{i+1}]/[{int(len(self.query_images_set)/query_batch)}]...')
                images_low=images_low[:,0,:,:,:]
                images_low=images_low.to(self.device)
                locations=locations[:,0,:]
                with torch.autocast('cuda', torch.float32):
                    features_low=self.vpr_model.model.encoder(images_low)
                    vectors_low=self.vpr_model.model.pool(features_low).detach().cpu()

                recall_score += self.vpr_examing(vectors_low, self.database_images_set['descriptors'],locations, self.database_images_set['locations'])
                del images_low, locations, features_low, vectors_low

            recall_rate = recall_score / len(self.query_images_set)

            for i,topk in enumerate(self.topk_nodes):
                self.writer.add_scalars(f'Recall rate/@{int(topk)}', {'trained model': recall_rate[0,i]}, iter_num)

def main(configs):
    root = configs['root']
    with open(join(root,'configs','test_dataset.yaml'), 'r') as f:
        data_split_config = yaml.safe_load(f)

    name=data_split_config['name']
    database_folder=data_split_config['database'][0]
    query_folder=data_split_config['query'][0]

    data_folders = {"database": join(root,'logs',name,database_folder), "query": join(root,'logs',name,query_folder)}

    vpr = VPRTester(root, data_folders, configs[...], configs[...])

    for iter_num in configs["num_repetitions"]:
        vpr.validation(iter_num)
    vpr.writer.close()
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/home/unav/Desktop/Resolution_Agnostic_VPR/configs/trainer.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    main(config)
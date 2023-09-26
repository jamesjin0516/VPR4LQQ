import argparse
import h5py
from os.path import join,exists
from os import listdir,makedirs
import json
from tqdm import tqdm
import yaml
import numpy as np

def process(image_folder,gt):
    nPosSample,nNegSample=1,5
    neighbor_file = h5py.File(join(image_folder,'neighbors.h5'), 'r')
    pitch_list=listdir(image_folder)
    pitch_list.remove('global_descriptor.h5')
    pitch_list.remove('neighbors.h5')
    for pitch in sorted(pitch_list):
        pitch_folder=join(image_folder,pitch)
        yaw_list=listdir(pitch_folder)
        for yaw in tqdm(sorted(yaw_list),total=len(yaw_list)):
            images_root=join(pitch_folder,yaw,'images')
            resolution_list=sorted(listdir(images_root))
            for resolution in resolution_list:
                resolution_folder=join(images_root,resolution)
                qp_list=listdir(resolution_folder)
                for qp in qp_list:
                    images_low_path=join(images_root,resolution,qp)
                    images_high_path=join(images_root,resolution_list[-1],'raw')

                    images_low=listdir(images_low_path)
                    for image in images_low:
                        name=f'{pitch}+{yaw}+{image}'
                        if name in neighbor_file:
                            image_high_path=join(images_high_path,image)
                            image_low_path=join(images_low_path,image)
                            positives_high,negtives_high=[],[]
                            positives_low,negtives_low=[],[]
                            image_locs=gt[image]

                            positives_locs,negtives_locs=[],[]
                            
                            ind=0
                            positives_pool=neighbor_file[name]['positives'][:]
                            negtives_pool=neighbor_file[name]['negtives'][:]
                            while len(positives_high)<nPosSample:
                                pitch_,yaw_,name_=positives_pool[ind].decode('utf-8').split('+')
                                positive_high=join(image_folder,pitch_,yaw_,'images',resolution_list[-1],'raw',name_)
                                positive_low=join(image_folder,pitch_,yaw_,'images',resolution,qp,name_)
                                if exists(positive_high) and exists(positive_low):
                                    positives_high.append(positive_high)
                                    positives_low.append(positive_low)
                                    positives_locs.append(gt[name_])
                                ind+=1
                                if ind==len(positives_pool)-1:
                                    break
                            ind=0
                            while len(negtives_high)<nNegSample:
                                pitch_,yaw_,name_=negtives_pool[ind].decode('utf-8').split('+')
                                negtive_high=join(image_folder,pitch_,yaw_,'images',resolution_list[-1],'raw',name_)
                                negtive_low=join(image_folder,pitch_,yaw_,'images',resolution,qp,name_)
                                if exists(negtive_high) and exists(negtive_low):
                                    negtives_high.append(negtive_high)
                                    negtives_low.append(negtive_low)
                                    negtives_locs.append(gt[name_])
                                ind+=1
                                if ind==len(negtives_pool)-1:
                                    break
                            if len(positives_high)==nPosSample and len(negtives_high)==nNegSample:
                                outf=join(pitch_folder,yaw,'neighbor',resolution,qp)
                                if not exists(outf):
                                    makedirs(outf)
                                locs=[image_locs]+positives_locs+negtives_locs
                                paths={'low':{'path':image_low_path,'positives':positives_low,'negtives':negtives_low},'high':{'path':image_high_path,'positives':positives_high,'negtives':negtives_high}}
                                save_dict={'paths':paths,'locs':locs}
                                save_path=join(outf,image.split('.')[0]+'.json')
                                with open(save_path,'w') as f:
                                    json.dump(save_dict,f)

def main(root,name):
    folders=listdir(root)
    folders=['Lighthouse6_0','Lighthouse6_1','Lighthouse6_2','Lighthouse6_3','Lighthouse6_4','Lighthouse6_5','Lighthouse6_6','Lighthouse6_7','Tandon4_0']
    if name=='unav':
        with open(join(root,'configs','measurement_scale.yaml'), 'r') as f:
            scales = yaml.safe_load(f)

    for image_folder in folders:
        if name=='unav':
            n=image_folder.split('_')[0]
            scale=scales[n]
            with open(join(root,'data',name,'utm',image_folder+'.json'),'r') as f:
                keyframes=json.load(f)['keyframes']
                gt={}
                for id, point in keyframes.items():
                    t_mp=point['trans']
                    gt[str(int(id)-1).zfill(5)+'.png'] = [t_mp[0]*scale, t_mp[1]*scale]

        process(join(root,'logs',name,image_folder),gt)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', type=str, default=None)
    parser.add_argument('-n', '--name', type=str, default=None)
    args = parser.parse_args()
    main(args.root,args.name)

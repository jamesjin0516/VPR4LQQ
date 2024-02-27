import argparse
import h5py
from os.path import join,exists
from os import listdir,makedirs
import json
from tqdm import tqdm
import yaml
import numpy as np
import shutil

def process(root,traj,gt,outf_):
    nPosSample,nNegSample=1,5
    image_folder=join(root,traj)

    neighbor_file = h5py.File(join(image_folder,'neighbors.h5'), 'r')
    if exists(join(image_folder,'neighbors.h5')):
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
                                        lists=positive_high.split('/')
                                        positive_high=lists[6]+'_'+lists[-1].replace('.png','')+'_pitch'+str(int(int(lists[7])/20+1))+'_yaw'+str(int(int(lists[8])/20+1))+'.png'
                                        positives_high.append(positive_high)
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
                                        lists=negtive_high.split('/')
                                        negtive_high=lists[6]+'_'+lists[-1].replace('.png','')+'_pitch'+str(int(int(lists[7])/20+1))+'_yaw'+str(int(int(lists[8])/20+1))+'.png'
                                        negtives_high.append(negtive_high)
                                        negtives_locs.append(gt[name_])
                                    ind+=1
                                    if ind==len(negtives_pool)-1:
                                        break
                                if len(positives_high)==nPosSample and len(negtives_high)==nNegSample:
                                    outf=join(pitch_folder,yaw,'neighbor',resolution,qp)
                                    if not exists(outf):
                                        makedirs(outf)
                                    locs=[image_locs]+positives_locs+negtives_locs
                                    save_dict={"positives": positives_high, "negatives": negtives_high, "locations": locs}

                                    dest_folder=join(outf_,resolution,qp)
                                    dest_image=dest_folder+'/'+traj+'_'+image.replace('.png','')+'_pitch'+str(int(int(pitch)/20+1))+'_yaw'+str(int(int(yaw)/20+1))+'.png'
                                    save_path=dest_folder+'/'+traj+'_'+image.replace('.png','')+'_pitch'+str(int(int(pitch)/20+1))+'_yaw'+str(int(int(yaw)/20+1))+'.json'
                                    
                                    if not exists(dest_folder):
                                        makedirs(dest_folder)
                                    shutil.move(image_low_path,dest_image)
                                    with open(save_path,'w') as f:
                                        json.dump(save_dict,f)

def main(root,name,outf):
    folders=sorted(listdir(join(root,'logs','unav')))
    # folders_=['ICT1_0','ICT1_1','ICT1_2','OP5_0','OP5_1','Outside1_0','370Jay9_1','ExhibitionHall1_0','ICT1_3','ICT2_3','ICT2_0','ICT2_1','ICT2_2','ICT3_0','ICT3_1','ICT3_2','ICT4_0','ICT4_1','ICT4_2','ICT4_3','Langone10_0','Langone10_1','Langone11_0','Langone11_1','Langone12_1','Langone17_0','Langone17_1',
    # 'Langone1_0','Langone2_0','Langone2_1','Langone5_0','Langone5_1','Langone6_0','Langone6_1','Langone8_0','Langone8_1','Langone9_0','Langone9_1','Library1_0','Library1_1','Library2_0','Library2_1','Library2_2',
    # 'MLC1_0','MLC1_1','MLC1_2','MLC2_0','MLC2_1','MLC2_2','MahidolHall1_0','MeetingHall1_0']
    
    # for folder in folders_:
    #     folders.remove(folder)

    # folders=['ICT1_2', 'ICT2_2', 'Ratchasuda3_1']
    folders=['ICT1_2','Ratchasuda3_1']
    with open(join(root,'configs','measurement_scale.yaml'), 'r') as f:
        scales = yaml.safe_load(f)
    
    for image_folder in folders:
        print(image_folder)
        n=image_folder.split('_')[0]
        if n in scales:
            scale=scales[n]
            with open(join(root,'data',name,'utm',image_folder+'.json'),'r') as f:
                keyframes=json.load(f)['keyframes']
                gt={}
                for id, point in keyframes.items():
                    t_mp=point['trans']
                    gt[str(int(id)-1).zfill(5)+'.png'] = [t_mp[0]*scale, t_mp[1]*scale]
            process(join(root,'logs',name),image_folder,gt,outf)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', type=str, default=None)
    parser.add_argument('-n', '--name', type=str, default=None)
    parser.add_argument('-o', '--outf', type=str, default=None)
    args = parser.parse_args()
    main(args.root,args.name,args.outf)

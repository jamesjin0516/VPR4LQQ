import numpy as np
import cv2
import os
from os.path import join, exists, dirname
import logging
import argparse
import yaml
import itertools
import sys
import json
from tqdm import tqdm

sys.path.append(dirname(sys.path[0]))
from tools.Compressor import compressor

# Equirectangular class definition for handling 360-degree images
class Equirectangular:
    RADIUS = 128

    # Initialize the Equirectangular object with the provided configuration
    def __init__(self, **config):
        slice_config = config['slice']
        pitch_config = slice_config['PITCH']
        yaw_config = slice_config['YAW']
        shape_config = slice_config['SHAPE']
        
        self.config = config
        self.pitch_list = [0] if pitch_config['num'] == 1 else np.linspace(-pitch_config['range'] / 2, pitch_config['range'] / 2, pitch_config['num'])
        self.yaw_list = np.linspace(0, 360, yaw_config['num'] + 1)[:-1]
        self.FOV = slice_config['FOV']
        self.height = shape_config['height']
        self.width = shape_config['width']
        self.wFOV = self.FOV
        self.hFOV = float(self.height) / self.width * self.wFOV
        self.c_x = (self.width - 1) / 2.0
        self.c_y = (self.height - 1) / 2.0
        self.wangle = (180 - self.wFOV) / 2.0
        self.hangle = (180 - self.hFOV) / 2.0
        
        self._compute_geometry()
        self.y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        self.z_axis = np.array([0.0, 0.0, 1.0], np.float32)

    # Compute the geometry of the equirectangular projection
    def _compute_geometry(self):
        self.w_len = 2 * self.RADIUS * np.sin(np.radians(self.wFOV / 2.0)) / np.sin(np.radians(self.wangle))
        self.h_len = 2 * self.RADIUS * np.sin(np.radians(self.hFOV / 2.0)) / np.sin(np.radians(self.hangle))
        self.w_interval = self.w_len / (self.width - 1)
        self.h_interval = self.h_len / (self.height - 1)

    # Create XYZ maps for the equirectangular projection
    def _create_xyz_maps(self):
        x_map = np.full((self.height, self.width), self.RADIUS, np.float32)
        y_map = np.tile((np.arange(0, self.width) - self.c_x) * self.w_interval, [self.height, 1])
        z_map = -np.tile((np.arange(0, self.height) - self.c_y) * self.h_interval, [self.width, 1]).T
        D = np.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)
        xyz = np.stack([(self.RADIUS / D * x_map), (self.RADIUS / D * y_map), (self.RADIUS / D * z_map)], axis=-1)
        return xyz
    
    # Apply rotation to the XYZ maps using input angles
    def _apply_rotation(self, xyz, THETA, PHI):
        R1, _ = cv2.Rodrigues(self.z_axis * np.radians(THETA))
        R2, _ = cv2.Rodrigues(np.dot(R1, self.y_axis) * np.radians(-PHI))
        xyz = xyz.reshape([self.height * self.width, 3]).T
        xyz = np.dot(R1, xyz)
        xyz = np.dot(R2, xyz).T
        return xyz

    # Get a perspective image from the equirectangular image using input angles
    def GetPerspective(self, image, THETA, PHI):
        equ_h, equ_w, _ = image.shape
        equ_cx, equ_cy = (equ_w - 1) / 2.0, (equ_h - 1) / 2.0
        xyz = self._create_xyz_maps()
        xyz = self._apply_rotation(xyz, THETA, PHI)

        lat = np.arcsin(xyz[:, 2] / self.RADIUS)
        lon = np.arctan2(xyz[:, 1], xyz[:, 0])
        lon, lat = np.degrees(lon).reshape(self.height, self.width), -np.degrees(lat).reshape(self.height, self.width)
        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90 * equ_cy + equ_cy

        persp = cv2.remap(image, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

        return persp

# Filter the low feature images
def low_feature_filter(image):
    orb = cv2.ORB_create()
    kp = orb.detect(image, None)
    kp, des = orb.compute(image, kp)
    if len(kp) > 100:
        return True
    return False

# Extract frames from video
def frame_extract(input_folder,keyframes_index,equ2pers,**config):
    pitch_list=equ2pers.pitch_list
    yaw_list=equ2pers.yaw_list
    Spatial_Res = [{k: v} for k, v in config['resolution'].items()]
    for PITCH in pitch_list:
        for YAW in tqdm(yaw_list,desc=f'extract frame at pitch {PITCH}',total=len(yaw_list)):
            work_folder=join(input_folder,str(int(PITCH)).zfill(3),str(int(YAW)).zfill(3))
            videos=[i for i in os.listdir(work_folder) if i.endswith('.mp4')]
            output_folder=join(work_folder,'images')
            if not exists(output_folder):
                os.makedirs(output_folder)
            for video in videos:
                video_name=join(work_folder,video)
                cap = cv2.VideoCapture(video_name)
                index=0
                while(cap.isOpened()):
                    ret, frame = cap.read()
                    if ret == True:
                        if (index in set(keyframes_index)) and low_feature_filter(frame):
                            if video=='raw.mp4':
                                for res in Spatial_Res:
                                    for k, v in res.items():
                                        width, height=v
                                        outf=join(output_folder,k,'raw')
                                        if not exists(outf):
                                            os.makedirs(outf)
                                        cv2.imwrite(join(outf,str(index).zfill(5)+'.png'),cv2.resize(frame,(height,width)))
                            else:
                                Resolution,qp=video.replace('.mp4','').split('_')
                                outf=join(output_folder,Resolution,qp)
                                if not exists(outf):
                                    os.makedirs(outf)
                                cv2.imwrite(join(outf,str(index).zfill(5)+'.png'),frame)
                    else: 
                        break
                    index+=1
                cap.release()

# Slice equirectangular video into perspective videos
def slice_data(input_path,output_path,gt_path,**config):
    equ2pers = Equirectangular(**config)
    pitch_list=equ2pers.pitch_list
    yaw_list=equ2pers.yaw_list
    with open(gt_path,'r') as f:
        keyframes_index=[int(i)-1 for i in list(json.load(f)['keyframes'])]
    for PITCH in pitch_list:
        for YAW in yaw_list:
            outf=join(output_path,str(int(PITCH)).zfill(3),str(int(YAW)).zfill(3))
            if not exists(outf):
                os.makedirs(outf)
            output_name=join(outf,'raw.mp4')
            if not exists(output_name):
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                out = cv2.VideoWriter(output_name, fourcc, 29.97, (equ2pers.width,equ2pers.height))
                cap = cv2.VideoCapture(input_path)
                if (cap.isOpened()== False): 
                    print("Error opening video stream or file")
                while(cap.isOpened()):
                # Capture frame-by-frame
                    ret, frame = cap.read()
                    if ret == True:
                        pers_frame=equ2pers.GetPerspective(frame,YAW,PITCH)
                        out.write(pers_frame)
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break
                    else: 
                        break
                # When everything done, release the video capture object
                cap.release()
                out.release()
                # Closes all the frames
                cv2.destroyAllWindows()
            compression(outf,**config['compression'])
    frame_extract(output_path,keyframes_index,equ2pers,**config['compression'])
    
# Compress the videos in the dataset
def compression(input_path,**config):
    input_video=join(input_path,'raw.mp4')
    QP_Range = config['QP']
    QP_Range = [str(i) for i in np.arange(QP_Range['min'], QP_Range['max'], QP_Range['interval'])]
    Spatial_Res = [{k: v} for k, v in config['resolution'].items()]
    for qp, res in itertools.product(QP_Range, Spatial_Res):
        for k, v in res.items():
            resolution=k
            output_path = join(input_path, f'{resolution}_{qp}.mp4')
            compressor(input_video, qp, v, output_path)

# Prepare the dataset by slicing and compressing videos
def prepare_data(opt,**config):
    root = config['root']
    
    ##### Slice database and query equirectangular video to perspective video
    input_path=join(root,'data',opt.database,'video',opt.video+'.mp4')
    output_path=join(root,'logs',opt.database,opt.video)
    gt_path=join(root,'data',opt.database,'utm',opt.video+'.json')

    slice_data(input_path,output_path,gt_path,**config['data'])

if __name__ == '__main__':
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/VPR_examiner.yaml')
    parser.add_argument('-v', '--video', type=str, default='Tandon4_0')
    parser.add_argument('-d', '--database', type=str, default='unav')
    opt = parser.parse_args()
    with open(opt.config, 'r') as f:
        config = yaml.safe_load(f)
    prepare_data(opt,**config)

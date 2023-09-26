import msgpack
from scipy.spatial.transform import Rotation as R
import numpy as np
import struct
import collections
import natsort
from os.path import join
import json

def slam2world(t, r):
    r = R.from_quat(r)
    return -np.matmul(r.as_matrix().transpose(), t)

class Colmap_GT():
    def __init__(self,path,T=np.ones((2,3)),rot_base=0):
        self.path=path
        self.T=T
        self.rot_base=rot_base

    def read_next_bytes(self,fid, num_bytes, format_char_sequence, endian_character="<"):
        data = fid.read(num_bytes)
        return struct.unpack(endian_character + format_char_sequence, data)

    def read_images_binary(self,path_to_model_file):
        images = {}
        with open(path_to_model_file, "rb") as fid:
            num_reg_images = self.read_next_bytes(fid, 8, "Q")[0]
            for image_index in range(num_reg_images):
                binary_image_properties = self.read_next_bytes(
                    fid, num_bytes=64, format_char_sequence="idddddddi")
                image_id = binary_image_properties[0]
                qvec = np.array(binary_image_properties[1:5])
                tvec = np.array(binary_image_properties[5:8])
                camera_id = binary_image_properties[8]
                image_name = ""
                current_char = self.read_next_bytes(fid, 1, "c")[0]
                while current_char != b"\x00":
                    image_name += current_char.decode("utf-8")
                    current_char = self.read_next_bytes(fid, 1, "c")[0]
                num_points2D = self.read_next_bytes(fid, num_bytes=8,
                                               format_char_sequence="Q")[0]
                x_y_id_s = self.read_next_bytes(fid, num_bytes=24 * num_points2D,
                                           format_char_sequence="ddq" * num_points2D)
                xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                       tuple(map(float, x_y_id_s[1::3]))])
                point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
                images[image_id] = {'id': image_id, 'qvec': qvec, 'tvec': tvec,
                                    'camera_id': camera_id, 'name': image_name,
                                    'xys': xys, 'point3D_ids': point3D_ids}
        return images

    def get_GT(self):
        images = self.read_images_binary(self.path)
        images = collections.OrderedDict(natsort.natsorted(images.items()))
        key = {}
        for id, point in images.items():
            trans = point["tvec"]
            rot = point['qvec']
            rot = [rot[1], rot[2], rot[3], rot[0]]
            pos = slam2world(trans, rot)
            t_fp = np.array([pos[0], pos[2], 1]).T
            t_mp = ((self.T @ t_fp).T).tolist()
            rot = -R.from_quat(rot).as_rotvec()[1] - self.rot_base
            key[point['name']] = [t_mp[0], t_mp[1], rot]
        return key

class OpenVSLAM_GT():
    def __init__(self,path):
        self.path=path

    def get_GT(self):
        with open(self.path, "r") as f:
            keyframes = json.load(f)['keyframes']
        key= {}
        for id, point in keyframes.items():
            t_mp=point['trans']
            rot=point['rot']
            key[id] = [t_mp[0], t_mp[1], rot]
        return key

def GT_pose(path,T=np.ones((2,3)),rot_base=0,pipeline='openvslam',testing_mode=False):
    if pipeline=='colmap':
        path=join(path+'-bm-sc','sfm_best_match_b30','models','0','images.bin')
        colmap_gt = Colmap_GT(path, T=T, rot_base=rot_base)
        return colmap_gt.get_GT()
    elif pipeline=='openvslam':
        # file='Testing_log.txt' if testing_mode else 'Mapping_log.txt'
        file='slam_data.json'
        path = join(path,file)
        # return OpenVSLAM_GT(path, T, rot_base)
        openvslam_gt = OpenVSLAM_GT(path)
        return openvslam_gt.get_GT()
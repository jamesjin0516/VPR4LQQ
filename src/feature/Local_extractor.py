import sys
import os
sys.path.append(os.path.join(sys.path[0],'..'))
from third_party.SuperPoint_SuperGlue.base_model import dynamic_load
from third_party.SuperPoint_SuperGlue import extractors,matchers
import numpy as np
import torch
from PIL import ImageOps

class Superpoint():
    conf = {
        'output': 'feats-superpoint-n4096-r1600',
        'model': {
            'name': 'superpoint',
            'nms_radius': 4,
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
        }}
    def __init__(self,device):
        Model_sp = dynamic_load(extractors, self.conf['model']['name'])
        self.local_feature_extractor=Model_sp(self.conf['model']).eval().to(device)
        self.device=device

    def prepare_data(self, image):
        image = np.array(ImageOps.grayscale(image)).astype(np.float32)
        image = image[None]
        data = torch.from_numpy(image / 255.).unsqueeze(0)
        return data

    def extract_local_features(self, image0):
        data0 = self.prepare_data(image0)
        pred0 = self.local_feature_extractor(data0.to(self.device))
        del data0
        torch.cuda.empty_cache()
        pred0 = {k: v[0].cpu().detach().numpy() for k, v in pred0.items()}
        if 'keypoints' in pred0:
            pred0['keypoints'] = (pred0['keypoints'] + .5) - .5
        pred0.update({'image_size': np.array([image0.size[0], image0.size[1]])})
        return pred0

class Superpoint_class():
    def __init__(self,device,**config):
        self.local_feature_extractor = loadModel(device,config['vpr']['local_feature']['path'])
        self.device=device

    def extract_local_features(self,image):
        params = {
            'out_num_points': 500,
            'patch_size': 5,
            'device': self.device,
            'nms_dist': 4,
            'conf_thresh': 0.015
        }
        image=image.to(self.device)
        out=self.local_feature_extractor(image)
        return out

class Local_extractor():
    device='cuda' if torch.cuda.is_available() else 'cpu'
    def __init__(self):
        self.superpoint=Superpoint(self.device)

    def superglue(self):
        conf_match = {
            'output': 'matches-superglue',
            'model': {
                'name': 'superglue',
                'weights': 'outdoor',
                'sinkhorn_iterations': 50,
            },
        }
        Model_sg = dynamic_load(matchers, conf_match['model']['name'])
        return Model_sg(conf_match['model']).eval()

    def extractor(self):
        return self.superpoint.extract_local_features

    def matcher(self):
        return self.superglue()
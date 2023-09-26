import cv2
import torch
import numpy as np
from PIL import Image
import sys
from os.path import join

sys.path.append(join(sys.path[0], '..'))
from dataset.Equirectangular_dataset import Equirectangular
from feature.Local_extractor import Local_extractor

class Geometric_verification():
    device='cuda' if torch.cuda.is_available() else 'cpu'
    def __init__(self):
        local_feature = Local_extractor()
        self.local_feature_extractor = local_feature.extractor()
        self.local_feature_matcher = local_feature.matcher().to(self.device)

    def process_perspective(self, image_path):
        image = cv2.imread(image_path)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        width, height = img.size
        scale = 640 / width
        newsize = (640, int(height * scale))
        img = img.resize(newsize)
        return img

    def geometric_verification(self, train_image, test_image):

        processor = self.process_perspective
        train_image = processor(train_image)
        train_feats = self.local_feature_extractor(train_image)

        processor = self.process_perspective
        test_image = processor(test_image)
        test_feats = self.local_feature_extractor(test_image)

        data = {}
        for k in test_feats.keys():
            data[k + '0'] = test_feats[k]
        for k in test_feats.keys():
            data[k + '1'] = train_feats[k].__array__()
        data = {k: torch.from_numpy(v)[None].float().to(self.device)
                for k, v in data.items()}
        data['image0'] = torch.empty((1, 1,) + tuple(test_feats['image_size'])[::-1])
        data['image1'] = torch.empty((1, 1,) + tuple(train_feats['image_size'])[::-1])
        pred = self.local_feature_matcher(data)
        matches = pred['matches0'][0].detach().cpu().short().numpy()
        pts0, pts1, lms = [], [], []
        for n, m in enumerate(matches):
            if (m != -1):
                pts0.append(test_feats['keypoints'][n].tolist())
                pts1.append(train_feats['keypoints'][m].tolist())
        try:
            pts0_ = np.int32(pts0)
            pts1_ = np.int32(pts1)
            F, mask = cv2.findFundamentalMat(pts0_, pts1_, cv2.RANSAC)
            valid = len(pts0_[mask.ravel() == 1])
            if valid > 20:
                return True
            else:
                return False
        except:
            return False

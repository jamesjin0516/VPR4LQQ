import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def read_coordinates(image_names):    # -> shape = (len(image_names), 2])
    pass


class TestDataset(Dataset):

    def __init__(self, image_folder, resolution):
        """
        - image_folder: path to query images folder (one level above individual resolutions folder)
        - resolution: specifies the resolution folder to choose for query images
        """
        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
        ])
        self.image_coordinates = read_coordinates(...)
        pass

    def __getitem__(self, index):
        pass
        return [image_high, image_low, location]
        # tensors, with shape [(1, #, #, #), (1, #, #, #), (1, 2)]

    def __len__(self):
        pass
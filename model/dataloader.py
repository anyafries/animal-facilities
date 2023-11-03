"""Data loader for CAFO Datasets"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image

URLS = {
    'poultry': {
        'train': 'https://drive.google.com/uc?id=1-8aZkr6H24dnQxkSaKjhCJBLaud1h7jR',
        'test': 'https://drive.google.com/uc?id=1hlkbPDuieDSQk2ItXsJH9ykXUxKANjlP',
        'val': 'https://drive.google.com/uc?id=1Foq8KplDbIRT1Tqrez9Qe8WUnQBUtlNh'
    },
    'dairy': {
        'train': 'https://drive.google.com/uc?id=1v6sAssgCp8ma6F8Jc8Qosh44A6aHti_1',
        'test': 'https://drive.google.com/uc?id=1Zlab_65D4l4z8wRiKQGxeZ8nS2slIIUO',
        'val': 'https://drive.google.com/uc?id=1OdamwiGvUSLk1aV75CdL2Ly1a-GluSK_'
    }

}

class CAFODataset(Dataset):
    def __init__(self, farm, split_set, transform, resolution=256, size=None,
                 path=None):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        if path is None:
          self.path = '/content/drive/MyDrive/animal_facilities/data/all_farms/most_recent/'+farm+"/"
        else:
          self.path = path

        if size is None:
          self.df = pd.read_csv(URLS[farm][split_set]).reset_index()#.set_index('idx')
        else:
          self.df = pd.read_csv(URLS[farm][split_set]).reset_index().sample(size, random_state=40)

        self.transform = transform
        # self.df = pd.read_csv(URLS[farm][split_set]).reset_index()
        self.filenames = np.asarray(self.df.iloc[:, 2])
        self.data_len = len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index].split('/')[-1]
        naip = Image.open(self.path+filename).convert('RGB')
        naip = self.transform(naip)
        label = self.df.iloc[index, 5]

        return naip, label

    def __len__(self):
        return self.data_len

    def plot(self, index, transform=None):
        if transform is None:
          naip, label = self.__getitem__(index)
        else:
          filename = self.filenames[index].split('/')[-1]
          naip = Image.open(self.path+filename).convert('RGB')
          naip = transform(naip)
          label = self.df.iloc[index, 5]

        img = naip.squeeze()
        plt.figure(figsize=(7,7))
        plt.imshow(img.permute(1, 2, 0))
        plt.title(f"Population: {label}")
        plt.axis('off')
        plt.show()
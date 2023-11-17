"""Data loader for CAFO Datasets"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image

URLS = {
    'poultry': {
        'train': 'https://drive.google.com/uc?id=1-jvxFZ9tTpL1mXGPqcuUc14GkgqMuGhu',
        'test': 'https://drive.google.com/uc?id=1-ePtRghQdojLypr5Pm96cpKx08V_7yvU',
        'val': 'https://drive.google.com/uc?id=1-dQ28FO2JBxAnzijjIm5_JoBVZZpejoK'
        # 'train': '/content/drive/MyDrive/CS 325B/splitting-files/poultry_train_bigger.csv',
        # 'test': '/content/drive/MyDrive/CS 325B/splitting-files/poultry_test_bigger.csv',
        # 'val': '/content/drive/MyDrive/CS 325B/splitting-files/poultry_val_bigger.csv'
    },
    'dairy': {
        'train': 'https://drive.google.com/uc?id=1-WfEWx1rw7ZRjnebeRbL66gfhh43yagX',
        'test': 'https://drive.google.com/uc?id=1-Rk7V_ToCot6zbS3sek6pgqDKrfWw-BO',
        'val': 'https://drive.google.com/uc?id=1-PF7o4Q_VCheCPFAKv-919Vuk5ZkqcU1'
        # 'train': '/content/drive/MyDrive/CS 325B/splitting-files/dairy_train_bigger.csv',
        # 'test': '/content/drive/MyDrive/CS 325B/splitting-files/dairy_test_bigger.csv',
        # 'val': '/content/drive/MyDrive/CS 325B/splitting-files/dairy_val_bigger.csv'
    },
    'beef': {
        'train': 'https://drive.google.com/uc?id=15HNJMqTHje2ALy52Ouui02eTMOtuNj1O',
        'test': 'https://drive.google.com/uc?id=1JdAXGG8aEwXqQ1vGcyLq4PdO9jHRF0yb',
        'val': 'https://drive.google.com/uc?id=1PMnYyC4v2XqhpgBt_23DvvNFG7EymgRf'
        # 'train': '/content/drive/MyDrive/CS 325B/splitting-files/beef_train_bigger.csv',
        # 'test': '/content/drive/MyDrive/CS 325B/splitting-files/beef_test_bigger.csv',
        # 'val': '/content/drive/MyDrive/CS 325B/splitting-files/beef_val_bigger.csv'
    },
    'mn': {
        'test': 'https://drive.google.com/uc?id=1mq-mZynSa3oS8m5q6QwpFQRTIpBNBQvC'
        # 'test': '/content/drive/MyDrive/CS 325B/8-same-models-more-data/mn-dairy-clean.csv'
    }
}

class CAFODataset(Dataset):
    def __init__(self, farm, split_set, transform, resolution=256,
                 augmented_data=True, path=None):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        if farm in ['dairy', 'poultry', 'beef']:
          self.path = 'data/gcs/data/all_farms/temporal/'
        elif farm == 'mn':
          self.path = 'data/gcs/data/mn/'
        else:
          raise NotImplementedError

        self.transform = transform
        self.df = pd.read_csv(URLS[farm][split_set])
        if not augmented_data:
          self.df = self.df[self.df['newest'] == True]
        self.filenames = np.asarray(self.df['filename'])
        self.data_len = len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]#.split('/')[-1]
        naip = Image.open(self.path+filename).convert('RGB')
        naip = self.transform(naip)
        label = self.df.iloc[index, 3]

        return naip, label

    def __len__(self):
        return self.data_len

    def plot(self, index, transform=None):
        if transform is None:
          naip, label = self.__getitem__(index)
        else:
          filename = self.filenames[index]#.split('/')[-1]
          naip = Image.open(self.path+filename).convert('RGB')
          naip = transform(naip)
          label = self.df.iloc[index, 3]

        img = naip.squeeze()
        plt.figure(figsize=(7,7))
        plt.imshow(img.permute(1, 2, 0))
        plt.title(f"Population: {label}")
        plt.axis('off')
        plt.show()
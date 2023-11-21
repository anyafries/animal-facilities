import os
import pandas as pd
import torch
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms

from model.dataloader import CAFODataset, URLS
from model.utils import Params, load_checkpoint
from unet_models.unet import DairyUnet, PoultryUnet, PoultryCAFODataset

def get_unet_features(farm, split_set, save_features=False):
    if farm == 'poultry':
        unet = PoultryUnet()
    else:
        unet = DairyUnet()

    # Which transforms to use?
    if farm == 'poultry':
        transform = transforms.Compose([
            transforms.CenterCrop(2048),
            transforms.ToTensor()])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(2016),
            transforms.ToTensor()])

    # Load the data
    print('Loading data for unet...')
    df_real = pd.read_csv(URLS[farm][split_set])
    if farm == 'poultry':
        df_set = PoultryCAFODataset(farm, split_set, transform)
    else:
        df_set = CAFODataset(farm, split_set, transform)
  
    # Extract features
    print('Extracting unet features...')
    df_out = df_real.copy(deep=True)
    df_out['unet_pixels'] = np.nan
    for i in range(len(df_set)):
        img = unet.infer(df_set[i][0].to('cuda'))
        df_set[i][0].detach()
        mask = unet.make_mask(img)
        num_white = unet.amount_white(mask)
        df_out.at[i, 'unet_pixels'] = num_white
    
    # Add back farm indexes
    df_out['idx'] = df_set.df['idx']

    # Save features
    if save_features:
        print('Saving unet features...')
        df_out.to_csv('unet_models/unet_'+farm+"_"+split_set+'.csv', index=False)
    else:
        return df_out

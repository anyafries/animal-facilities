import pandas as pd
import numpy as np
import os

from torchvision import transforms

from model.dataloader import CAFODataset, URLS
from model.unet import DairyUnet, PoultryUnet

def get_unet_features(farm, split_set, save_features=False):
    if farm in ['mn']:
        model_dir = 'experiments/unet_features/dairy'
    elif farm in ['kt', 'kt_uncentered', 'og', 'sc', 'ms', 'ks', 'az']:
        model_dir = 'experiments/unet_features/poultry'
    else:
        model_dir = 'experiments/unet_features/' + farm
    filename = os.path.join(model_dir, 'unet_'+farm+"_"+split_set+'.csv')

    if farm in ['poultry', 'kt', 'kt_uncentered', 'og', 'sc', 'ms', 'ks', 'az']:
        unet = PoultryUnet()
    else:
        unet = DairyUnet()

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # Which transforms to use?
        if farm in ['poultry', 'kt', 'kt_uncentered', 'og', 'sc', 'ms', 'ks', 'az']:
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
        if farm in ['poultry', 'kt', 'kt_uncentered', 'og', 'sc', 'ms', 'ks', 'az']:
            df_set = CAFODataset(farm, split_set, transform, poultry_unet=True)
        else:
            df_set = CAFODataset(farm, split_set, transform)
    
        # Extract features
        print('Extracting unet features...')
        df_out = df_real.copy(deep=True)
        df_out['unet_pixels'] = np.nan
        for i in range(len(df_set)):
            if i % 100 == 0:
                print(f"\t{i} / {len(df_set)}")
            # check if the img has 4 channels
            img = df_set[i][0]
            if img.shape[0] != 4:
                num_white = np.nan
            else:
                img = unet.infer(img.to('cuda'))
                df_set[i][0].detach()
                mask = unet.make_mask(img)
                num_white = unet.amount_white(mask)
            df_out.at[i, 'unet_pixels'] = num_white
        
        # Add back farm indexes
        df_out['idx'] = df_set.df['idx']

        # Save features
        if save_features:
            print('Saving unet features...')
            df_out.to_csv(filename, index=False)
        
        return df_out

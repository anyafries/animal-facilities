import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import cv2

class HOGLayer(nn.Module):
    def __init__(self, nbins=10, pool=8, max_angle=math.pi, stride=1, padding=1, dilation=1):
        super(HOGLayer, self).__init__()
        self.nbins = nbins
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pool = pool
        self.max_angle = max_angle
        mat = torch.FloatTensor([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        mat = torch.cat((mat[None], mat.t()[None]), dim=0)
        self.register_buffer("weight", mat[:,None,:,:])
        self.pooler = nn.AvgPool2d(pool, stride=pool, padding=0, ceil_mode=False, count_include_pad=True)

    def forward(self, x):
        with torch.no_grad():
            gxy = F.conv2d(x, self.weight, None, self.stride,
                            self.padding, self.dilation, 1)
            #2. Mag/ Phase
            mag = gxy.norm(dim=1)
            norm = mag[:,None,:,:]
            phase = torch.atan2(gxy[:,0,:,:], gxy[:,1,:,:])

            #3. Binning Mag with linear interpolation
            phase_int = phase / self.max_angle * self.nbins
            phase_int = phase_int[:,None,:,:]

            n, c, h, w = gxy.shape
            out = torch.zeros((n, self.nbins, h, w), dtype=torch.float, device=gxy.device)
            out.scatter_(1, phase_int.floor().long()%self.nbins, norm)
            out.scatter_add_(1, phase_int.ceil().long()%self.nbins, 1 - norm)

            return self.pooler(out)


def extract_hog(image, hog):
    transform = transforms.Grayscale()
    image = transform(image[0:3])
    image = image.unsqueeze(0)
    # print(image.shape)
    y = hog(image)
    return y


def extract_histogram(image, plot=False):
    image = image.numpy() * 255

    colors = ("red", "green", "blue")
    labels = ("red", "green", "blue")

    # create the histogram plot, with three lines, one for each color
    if plot:
        plt.figure()
        plt.xlim([0, 256])

    histogram = [0] * 3
    for channel_id, color in enumerate(colors):
        histogram[channel_id], bin_edges = np.histogram(
            image[channel_id, :, :], bins=256, range=(0, 256)
        )
        if plot:
            plt.plot(bin_edges[0:-1], histogram[channel_id], color=color, label=labels[channel_id])

    if plot:
        plt.title("Color Histogram")
        plt.xlabel("Color value")
        plt.ylabel("Pixel count")
        plt.legend()

    histogram = np.array(histogram)
    return histogram


# get the mean and std given a histogram and bin values
def hist_mean(histogram, bin_vals = np.arange(256)):
    return np.sum(histogram * bin_vals) / np.sum(histogram)


def hist_std(histogram, bin_vals = np.arange(256)):
    mean = np.sum(histogram * bin_vals) / np.sum(histogram)
    squared_diff = (bin_vals - mean)**2
    weighted_squared_diff = squared_diff * histogram
    sum_weighted_squared_diff = np.sum(weighted_squared_diff)
    std_dev = np.sqrt(sum_weighted_squared_diff / np.sum(histogram))
    return std_dev


# get the min, LQ, median, UQ, max given a histogram and bin values
def hist_summ(histogram, bin_vals = np.arange(256)):
  values = np.array(bin_vals)
  quantiles = np.array([0,0.25,0.5,0.75,1])
  sample_weight = np.array(histogram)

  weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
  weighted_quantiles /= np.sum(sample_weight)
  return np.interp(quantiles, weighted_quantiles, values)


def get_cols():
    hog_means = ['hog_mean_0', 'hog_mean_1', 'hog_mean_2']
    hog_stds = ['hog_std_0', 'hog_std_1', 'hog_std_2']
    rgb_means = ['r_mean', 'g_mean', 'b_mean']
    rgb_stds = ['r_std', 'g_std', 'b_std']
    hist0 = ['hog_hist_0_min', 'hog_hist_0_LQ', 'hog_hist_0_median', 'hog_hist_0_UQ', 'hog_hist_0_max']
    hist1 = ['hog_hist_1_min', 'hog_hist_1_LQ', 'hog_hist_1_median', 'hog_hist_1_UQ', 'hog_hist_1_max']
    hist2 = ['hog_hist_2_min', 'hog_hist_2_LQ', 'hog_hist_2_median', 'hog_hist_2_UQ', 'hog_hist_2_max']
    r = ['r_min', 'r_LQ', 'r_median', 'r_UQ', 'r_max']
    g = ['g_min', 'g_LQ', 'g_median', 'g_UQ', 'g_max']
    b = ['b_min', 'b_LQ', 'b_median', 'b_UQ', 'b_max']
    all_cols = ['idx'] + hog_means + hog_stds + rgb_means + rgb_stds + hist0 + hist1 + hist2 + r + g + b
    return all_cols


def extract_simple_features(dataset):
    df = []
    hog_model = HOGLayer(nbins=3, pool=2)
    for range(len(dataset)):
        image, _ = dataset[i]
        hog = extract_hog(image, hog_model)
        histogram = extract_histogram(image)

        hog0 = hog[0][0].flatten()
        hog1 = hog[0][1].flatten()
        hog2 = hog[0][2].flatten()
        rgb = histogram

        hist_0 = np.histogram(hog0, bins=np.linspace(0, 1, 11), density=True)[0]
        hist_1 = np.histogram(hog1, bins=np.linspace(0, 1, 11), density=True)[0]
        hist_2 = np.histogram(hog2, bins=np.linspace(0, 1, 11), density=True)[0]
        h0_summ = np.percentile(hog0, [0, 25, 50, 75, 100])
        h1_summ = np.percentile(hog1, [0, 25, 50, 75, 100])
        h2_summ = np.percentile(hog2, [0, 25, 50, 75, 100])

        info = np.array([
        i,
        hog0.mean().item(), hog1.mean().item(), hog2.mean().item(),
        hog0.std().item(), hog1.std().item(), hog2.std().item(),
        hist_mean(rgb[0].flatten()), hist_mean(rgb[1].flatten()), hist_mean(rgb[2].flatten()),
        hist_std(rgb[0].flatten()), hist_std(rgb[1].flatten()), hist_std(rgb[2].flatten()),
        ])
        info = np.hstack([info, h0_summ, h1_summ, h0_summ,
                        hist_summ(rgb[0]), hist_summ(rgb[1]), hist_summ(rgb[2])])

        df.append(info)

    all_cols = get_cols()
    df = pd.DataFrame(df, columns=all_cols)
    return df


def get_simple_features(model_dir, farm, split_set, save_features=False):
    if farm == 'mn':
        model_dir = model_dir + '/dairy'
    else:
        model_dir = model_dir + '/' + farm
    
    json_path = os.path.join(model_dir, 'params.json')

    transform = transforms.Compose([
        transforms.CenterCrop(2048),
        transforms.ToTensor(),
    ])
    df_in = CAFODataset(farm, split_set, transform)
    
    print('Extracting simple features...')
    df_out = extract_simple_features(df_in)
    df_out['idx'] = df_in.df['idx']
    
    # Save features
    if save_features:
        print('Saving simple features...')
        df_out.to_csv(os.path.join(model_dir, 'simple_'+farm+"_"+split_set+'.csv'), index=False)
    else:
        return df_out
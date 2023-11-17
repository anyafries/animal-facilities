import os
import pandas as pd
import torch

from torch.utils.data import DataLoader
from torchvision import transforms

from model.dataloader import CAFODataset
from model.utils import Params, load_checkpoint
from model.models import get_resnet50


def extract_deep_features(model, layer, data_loader, device='cpu'):
    outputs = []
    df = pd.DataFrame()

    def hook_fn(module, input, output):
        outputs.append(output)

    hook = layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)

            outputs = []
            preds = model(images)

            features_np = outputs[0].cpu().numpy()
            df = pd.concat([df, pd.DataFrame(features_np)], ignore_index=True)
            print(df.shape)

    hook.remove()
    return df


def get_resnet_features(model_dir, farm, split_set, save_features=False):
    if farm == 'mn':
        model_dir = model_dir + '/dairy'
    else:
        model_dir = model_dir + '/' + farm
    
    json_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if params.model == 'resnet':
        model = get_resnet50().to(device)
    
    # Load weights
    restore_path = os.path.join(model_dir, 'best_model.pth.tar')
    load_checkpoint(restore_path, model)

    # Which transforms to use?
    transform = transforms.Compose([
        transforms.CenterCrop(2048),
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the data
    print('Loading data for resnet...')
    df_in = CAFODataset(farm, split_set, transform)
    dataloader = DataLoader(df_in, batch_size=params.batch_size, 
                            shuffle=False, num_workers=2, prefetch_factor=4)
  
    # Extract features
    layer = model.fc[1]
    print('Extracting resnet features...')
    df_out = extract_deep_features(model, layer, dataloader, device)

    # Add back farm indexes
    df_out['idx'] = df_in.df['idx']

    # Save features
    if save_features:
        print('Saving resnet features...')
        df_out.to_csv(os.path.join(model_dir, 'resnet_'+farm+"_"+split_set+'.csv'), index=False)
    else:
        return df_out
import argparse
import os
import pandas as pd
import torch

from torch.utils.data import DataLoader
from torchvision import transforms

from model.dataloader import CAFODataset
from model.models import get_resnet50 #, get_densenet121
from model.utils import Params, load_checkpoint

def extract_features(model, layer, data_loader, device='cpu'):

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


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model', 
                    help="Directory containing weights to reload from.json")


if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if params.model == 'resnet':
        model = get_resnet50().to(device)
    elif params.model == 'densenet':
        # model = get_densenet121(params).to(device)
        raise NotImplementedError
    
    # Load weights
    restore_path = os.path.join(args.model_dir, 'best_model.pth.tar')
    load_checkpoint(restore_path, model)

    # Which transforms to use?
    if params.model == 'resnet':
        transform = transforms.Compose([
            transforms.CenterCrop(2048),
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif params.model == 'densenet':
        raise NotImplementedError

    # Load the data
    print('Loading data...')
    train_set = CAFODataset('dairy', 'train', transform)
    val_set = CAFODataset('dairy', 'val', transform)
    test_set = CAFODataset('dairy', 'test', transform)

    train_loader = DataLoader(train_set, batch_size=params.batch_size, 
                              shuffle=True, num_workers=2, prefetch_factor=4)
    val_loader = DataLoader(val_set, batch_size=params.batch_size, 
                            shuffle=False, num_workers=2, prefetch_factor=4)
    test_loader = DataLoader(test_set, batch_size=params.batch_size, 
                             shuffle=True, num_workers=2, prefetch_factor=4)

    # Extract features
    layer = model.fc[1]
    print('Extracting train features...')
    train_df = extract_features(model, layer, train_loader, device)
    print('Extracting val features...')
    val_df = extract_features(model, layer, val_loader, device)
    print('Extracting test features...')
    test_df = extract_features(model, layer, test_loader, device)

    # Save features
    print('Saving features...')
    train_df.to_csv(os.path.join(args.model_dir, 'train_features.csv'), index=False)
    val_df.to_csv(os.path.join(args.model_dir, 'val_features.csv'), index=False)
    test_df.to_csv(os.path.join(args.model_dir, 'test_features.csv'), index=False)
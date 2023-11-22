"""Train the model"""

import argparse
import logging
import os
import random
import torch

from torch.utils.data import DataLoader
from torchvision import transforms

from model.dataloader import CAFODataset
from model.models import get_resnet50 #, get_densenet121
from model.training import train_and_evaluate, EarlyStopper
from model.utils import Params, set_logger


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/resnet_features',
                    help="Experiment directory containing params.json")
parser.add_argument('--farm', default='dairy',
                    help="Which farm? dairy, poultry, or beef")
parser.add_argument('--restore_from', default=False,
                    help="Optional, directory or file containing weights to reload before training")


if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    torch.manual_seed(230)

    # Load the parameters from json file
    args = parser.parse_args()
    model_dir = args.model_dir + '/' + args.farm
    json_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwritting
    model_dir_has_best_weights = os.path.isdir(
        os.path.join(model_dir, "best_weights"))
    overwritting = model_dir_has_best_weights and args.restore_from is None
    assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

    # Set the logger
    set_logger(os.path.join(model_dir, 'train.log'))
    logging.info(f"Starting training for {args.farm} @ {args.model_dir}...")

    # Which transforms to use?
    if params.model == 'resnet':
        transform = transforms.Compose([
            # rotation
            transforms.RandomRotation(90),
            transforms.CenterCrop(2048), # add first to be consistent with previous work
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # transforms.Normalize(mean=[0,0,0], std=[1,1,1])
        ])
        test_transform = transforms.Compose([
            transforms.CenterCrop(2048),
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # transforms.Normalize(mean=[0,0,0], std=[1,1,1])
        ])
    elif params.model == 'densenet':
        raise NotImplementedError

    # Create the dataloaders
    logging.info("Loading the datasets...")
    train_set = CAFODataset(args.farm, 'train', transform)
    val_set = CAFODataset(args.farm, 'val', test_transform, augmented_data=False)
    # test_set = CAFODataset(args.farm, 'test', test_transform)

    train_loader = DataLoader(train_set, batch_size=params.batch_size, 
                              shuffle=True, num_workers=2, prefetch_factor=4)
    val_loader = DataLoader(val_set, batch_size=params.batch_size, 
                            shuffle=False, num_workers=2, prefetch_factor=4)
    # test_loader = DataLoader(test_set, batch_size=params.batch_size, 
    #                          shuffle=True, num_workers=2, prefetch_factor=4)
    dataloaders = {'train': train_loader, 'val': val_loader}

    # Define the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Creating the model, send to {device}...")
    if params.model == 'resnet':
        model = get_resnet50().to(device)
    if params.model == 'densenet':
        # model = get_densenet121()
        raise NotImplementedError

     # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=params.step_size, gamma=params.gamma)
    early_stopper = EarlyStopper(patience=params.patience, min_delta=params.min_delta)

    # Train the model
    logging.info("Call the training loop...")
    train_and_evaluate(model, criterion, optimizer, scheduler, dataloaders,
                       device, early_stopper, model_dir, params, 
                       args.restore_from)
    

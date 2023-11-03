"""Train the model"""

import argparse
import logging
import os
import random
import torch

from torchvision import transforms

from model.dataloader import CAFODataset
from model.training import train_and_evaluate
from model.utils import Params, set_logger, EarlyStopper
from model.models import get_resnet50 #, get_densenet121


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Experiment directory containing params.json")
# parser.add_argument('--data_dir', default='data/64x64_SIGNS',
#                     help="Directory containing the dataset")
parser.add_argument('--restore_from', default=False,
                    help="Optional, directory or file containing weights to reload before training")


if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    torch.set_random_seed(230)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwritting
    model_dir_has_best_weights = os.path.isdir(
        os.path.join(args.model_dir, "best_weights"))
    overwritting = model_dir_has_best_weights and args.restore_from is None
    assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Which transforms to use?
    if params.model == 'resnet':
        transform = transforms.Compose([
            transforms.CenterCrop(2048), # add first to be consistent with previous work
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
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
        assert False, "Densenet not implemented yet"

    # Create the dataloaders
    logging.info("Loading the datasets...")
    train_set = CAFODataset('dairy', 'train', transform)
    val_set = CAFODataset('dairy', 'val', test_transform)
    # test_set = CAFODataset('dairy', 'test', test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=params.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=params.batch_size, shuffle=False, num_workers=4)
    # test_loader = torch.utils.data.DataLoader(
    #     test_set, batch_size=params.batch_size, shuffle=False, num_workers=4)
    dataloaders = {'train': train_loader, 'val': val_loader}

    # Define the model
    logging.info("Creating the model...")
    if params.model == 'resnet':
        model = get_resnet50()
    if params.model == 'densenet':
        # model = get_densenet121()
        assert False, "Densenet not implemented yet"

    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=params.step_size, gamma=params.gamma)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    early_stopper = EarlyStopper()

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, criterion, optimizer, scheduler, dataloaders,
                       device, args.model_dir, params, args.restore_from)
    

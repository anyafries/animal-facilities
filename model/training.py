"""Utility functions for evaluating the model."""

import logging
import numpy as np

from model.utils import save_checkpoint, load_checkpoint, save_dict_to_json
from model.utils import save_and_reset_bn_statistics, restore_bn_statistics
from torch.utils.tensorboard import SummaryWriter

class EarlyStopper:
    def __init__(self, patience=4, min_delta=100):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

def train_and_evaluate(model, criterion, optimizer, scheduler, dataloaders, 
                       device, early_stopper, model_dir, params, 
                       restore_from=None):
    begin_epoch = 0

    if restore_from is not None:
        logging.info(f"Restoring parameters from {restore_from}")
        begin_epoch = load_checkpoint(restore_from, model, optimizer)

    best_loss = np.Inf
    losses = {'train': [], 'val': []}
    mae = {'train': [], 'val': []}

    # For tensorboard
    writer = SummaryWriter(log_dir=f"{model_dir}/logs")

    for epoch in range(begin_epoch, begin_epoch + params.num_epochs):
        current_lr = scheduler.get_last_lr()[0]

        if params.cutoff > 0 and epoch == params.cutoff:
            for param in model.layer4.parameters():
                param.requires_grad = False

        for phase in ['train', 'val']:
            dataloader = dataloaders[phase]

            if phase == 'train': model.train()
            else: model.eval()

            squared_errors = []
            errors = []

            for inputs, labels in dataloader:
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()

                if phase == 'train':
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), labels)
                    loss.backward()
                    optimizer.step()
                if phase == 'val':
                    outputs = model(inputs)
                    bn_statistics = save_and_reset_bn_statistics(model)
                    loss = criterion(outputs.squeeze(), labels)
                    restore_bn_statistics(model, bn_statistics)

                squared_error = (outputs.squeeze() - labels)**2
                squared_errors.extend(squared_error.cpu().detach().numpy())
                errors.extend((outputs.squeeze() - labels).cpu().detach().numpy())

            if phase == 'train':
                scheduler.step()

            epoch_loss = np.sqrt(np.mean(squared_errors))
            losses[phase].append(epoch_loss)
            epoch_mae = np.mean(np.abs(errors))
            mae[phase].append(epoch_mae)

            if phase == 'val' and epoch_loss < best_loss and epoch_loss < 400:
                best_loss = epoch_loss
                # save the model
                filename = f"{model_dir}/best_model.pth.tar"
                logging.info(f"Saving the model at epoch {epoch+1}, saving in {filename}")
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, filename=filename)
                # save best eval metrics in a json file in the model directory
                metrics = {'epoch': epoch + 1, 'loss': epoch_loss, 'mae': epoch_mae}
                save_dict_to_json(metrics, f"{model_dir}/best_model_eval_metrics.json")

        train_loss, val_loss = losses['train'][-1], losses['val'][-1]
        train_mae, val_mae = mae['train'][-1], mae['val'][-1]
        logging.info(f'Epoch: {begin_epoch+epoch+1}/{begin_epoch+params.num_epochs} \tTrain Loss: {train_loss:.1f} \tVal Loss: {val_loss:.1f} \t\tLR: {current_lr:.6f}')
        
        # For tensorboard
        writer.add_scalar(f'lr', current_lr, epoch)
        writer.add_scalar(f'train/loss', train_loss, epoch)
        writer.add_scalar(f'val/loss', val_loss, epoch)
        writer.add_scalar(f'train/mae', train_mae, epoch)
        writer.add_scalar(f'val/mae', val_mae, epoch)

        if early_stopper.early_stop(val_loss):
          print("stopping early!")
          load_checkpoint(filename)
          break

    load_checkpoint(filename)

    return model, losses, mae
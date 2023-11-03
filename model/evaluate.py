"""Utility functions for evaluating the model."""

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import linregress


def plot_preds(true_value, predicted_value, title):
    plt.figure(figsize=(10,10))
    if type(predicted_value[0]) == "list":
        for i in range(len(predicted_value)):
            r_val = np.corrcoef(true_value, predicted_value[i])[0,1]
            plt.scatter(true_value, predicted_value[i], label=f'r = {r_val:.2f}')
    else:
        r_val = np.corrcoef(true_value, predicted_value)[0,1]
        plt.scatter(true_value, predicted_value, label=f'r = {r_val:.2f}')

    p1 = max(max(predicted_value), max(true_value))
    p2 = min(min(predicted_value), min(true_value))
    plt.plot([p1, p2], [p1, p2], '--')
    plt.xlabel('True Population Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_loss(loss, mae, ymax=4000):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(loss['train'], label='train')
    ax[0].plot(loss['val'], label='val')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_ylim(0, ymax)
    ax[0].legend(frameon=False)

    ax[1].plot(mae['train'], label='train')
    ax[1].plot(mae['val'], label='val')
    ax[1].set_title('MAE')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('MAE')
    ax[1].set_ylim(0, ymax)
    ax[1].legend(frameon=False)

    plt.show()


def evaluate(model, data_loader, device, print_out=True, return_out=False, 
             plot=False, title='Predicted vs. True'):
    with torch.no_grad():
        y_pred = []
        y_true = []
        for images, labels in data_loader:
            images = images.to(device).float()
            outputs = model(images)
            y_pred.extend(outputs.cpu().numpy())
            y_true.extend(labels.numpy())
        y_pred = np.asarray(y_pred).squeeze()
        y_true = np.asarray(y_true)
        rmse = np.sqrt(np.mean((y_pred - y_true)**2))
        mae = np.mean(np.abs(y_pred - y_true))
        r2 = np.corrcoef(y_pred, y_true)[0,1]**2
        slope, intercept, r_value, p_value, std_err = linregress(y_true, y_pred)

        if print_out:
            print(f"RMSE: {rmse:.2f} \tMAE: {mae:.2f} \tR2: {r2:.2f} \tSlope: {slope:.4f} \tIntercept {intercept:.4f}")
        if plot:
            plot_preds(y_true, y_pred, title)
        if return_out:
            return rmse, mae, r2, slope, intercept
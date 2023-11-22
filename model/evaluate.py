"""Utility functions for evaluating the model."""

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


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


def get_xy(dataset, resnet_cols):
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
    unet_pixels = ['unet_pixels']
    resnet = resnet_cols

    var_cols = hog_means + hog_stds + rgb_means + rgb_stds + hist0 + hist1 + hist2 + r + g + b + unet_pixels + resnet

    Y = dataset['population']
    X = dataset[var_cols]
    return X, Y


def get_evaluation(true, pred):
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    mape = mean_absolute_percentage_error(true, pred)
    slope, intercept, r_value, p_value, std_err = linregress(true, pred)
    return rmse, mae, mape, slope, intercept
    

def test_model(model, train, val, resnet_cols, model_name,
               print_out=False, plot=False):
    X, Y = get_xy(train, resnet_cols)
    model.fit(X, Y)
    Y_pred = model.predict(X)
    if plot:
        plot_preds(Y, Y_pred, f'Train preds: {model_name}')

    X, Y = get_xy(val, resnet_cols)
    Y_pred = model.predict(X)
    if plot:
        plot_preds(Y, Y_pred, f'Val preds: {model_name}')

    rmse, mae, mape, slope, intercept = get_evaluation(Y, Y_pred)
    r2 = model.score(X, Y)

    if print_out:
        print(f"RMSE: {rmse:.2f} \tMAE: {mae:.2f} \tMAPE: {mape:.2f} \tMAE: {mae:.2f} \tR2: {r2:.2f} \tSlope: {slope:.4f} \tIntercept {intercept:.4f}")
    
    out = {
        'Model' : model_name,
        'RMSE' : rmse,
        'MAE' : mae,
        'MAPE' : mape,
        'R^2' : r2,
        'Slope' : slope,
        'Intercept' : intercept
    }
    # print(out)
    return out
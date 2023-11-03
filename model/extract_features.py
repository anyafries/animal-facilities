import torch
import pandas as pd

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


# TODO: write an argument parser for this and make it a script
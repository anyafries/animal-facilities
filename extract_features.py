import argparse

from model.extract_resnet_features import get_resnet_features
from model.extract_simple_features import get_simple_features
from unet_models.extract_unet_features import get_unet_features


parser = argparse.ArgumentParser()
parser.add_argument('--resnet_dir', default='experiments/bigger_model_v1', 
                    help="Directory containing weights to reload from.json")
parser.add_argument('--simple_dir', default='experiments/simple_features', 
                    help="Directory containing weights to reload from.json")
parser.add_argument('--farm', default='dairy',
                    help="Which farm? dairy, poultry, or beef")
parser.add_argument('--split_set', default='test',
                    help="Which split? train, val, or test")

if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    resnet_dir = args.resnet_dir
    farm = args.farm
    split_set = args.split_set
    
    # Get features
    print('Getting features...')
    # TODO: 
    df_simple = get_simple_features(simple_dir, farm, split_set)
    df_unet = get_unet_features(farm, split_set)
    df_resnet = get_resnet_features(resnet_dir, farm, split_set)
    
    # TODO: merge features
    df_out = pd.concat((df_simple, df_resnet, df_unet[['unet_pixels']]), axis=1)

    # Save features
    print('Saving features...')
    df_out.to_csv(f"features/{farm}_{split_set}.csv", index=False)

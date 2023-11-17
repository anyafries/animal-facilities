import argparse

from model.extract_resnet_features import get_resnet_features


parser = argparse.ArgumentParser()
parser.add_argument('--resnet_dir', default='experiments/bigger_model_v1', 
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
    # df_simple = 
    # df_unet = 
    df_resnet = get_resnet_features(resnet_dir, farm, split_set)
    
    # TODO: merge features
    df_out = df_resnet 

    # Save features
    print('Saving features...')
    df_out.to_csv(f"features/{farm}_{split_set}.csv", index=False)
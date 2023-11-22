import argparse
import pandas as pd

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

from model.evaluate import test_model

parser = argparse.ArgumentParser()
parser.add_argument('--features_dir', default='features', 
                    help="Directory containing extracted features")
parser.add_argument('--farm', default='dairy',
                    help="Which farm? dairy, poultry, or beef")
parser.add_argument('--split_set', default='test',
                    help="Which split to evaluate on? val, or test")


def get_results(FARM, train_simple, val_simple, resnet_cols,
                multiple_train=False, multiple_val=False):
    if not multiple_train:
        train_simple = train_simple[train_simple['newest'] == True]
    if not multiple_val:
        val_simple = val_simple[val_simple['newest'] == True]

    results = pd.DataFrame()
    models = {
        # 'Linear Regression' : LinearRegression(),
        'Lasso' : Lasso(alpha=1),
        'Ridge' : Ridge(alpha=1),
        'Decision Tree' : DecisionTreeRegressor(max_depth=4),
        'Gradient Boosting' : GradientBoostingRegressor(random_state=0),
        'Random Forest' : RandomForestRegressor(max_depth=4, random_state=0),
        'AdaBoost' : AdaBoostRegressor(random_state=0, n_estimators=100),
        # 'Elastic Net' : ElasticNet(random_state=0),
        # 'Multi-Layer Perceptron' : MLPRegressor(),
        # 'KNN' : KNeighborsRegressor(n_neighbors=5)
    }

    for name, model in models.items():
        metrics = test_model(model, train=train_simple, val=val_simple, resnet_cols=resnet_cols, model_name=name, var_names="simple + deep features")
        metrics_df = pd.DataFrame([metrics])
        results = pd.concat([results, metrics_df], ignore_index=True)

    return results.round(2)


if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    farm = args.farm
    split_set = args.split_set
    
    # Get features
    print('Getting features...')
    train = pd.read_csv(f"features/{farm}_train.csv")
    val = pd.read_csv(f"features/{farm}_{split_set}.csv")
    # resnet_cols = # TODO

    # Get results
    print('Getting results...')
    res = get_results(farm, train, val, resnet_cols)
    print(res)
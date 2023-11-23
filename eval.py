import argparse
import pandas as pd

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

from model.evaluate import test_model, get_preds

parser = argparse.ArgumentParser()
parser.add_argument('--farm', default='dairy',
                    help="Which farm? dairy, poultry, or beef")
parser.add_argument('--split_set', default='test',
                    help="Which split to evaluate on? val, or test")
parser.add_argument('--only_preds', default=False, 
                    help="Only get the predictions for the outcome variable?")


def get_results(FARM, train, val, resnet_cols, only_preds=False):#,
                # multiple_train=False, multiple_val=False):
    # if not multiple_train:
    #     train = train[train['newest'] == True]
    # if not multiple_val:
    #     val = val[val['newest'] == True]

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
    
    if only_preds:
        results['idx'] = val['idx']
        for name, model in models.items():
            preds = get_preds(model, train, val, resnet_cols)
            results[name] = preds
    else:
        for name, model in models.items():
            metrics = test_model(model, train=train, val=val, 
                                 resnet_cols=resnet_cols, model_name=name)
            metrics_df = pd.DataFrame([metrics])
            results = pd.concat([results, metrics_df], ignore_index=True)
        results = results.round(2)
        print(results)

    return results


if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    only_preds = args.only_preds == 'True'
    farm = args.farm
    train_farm = farm
    if farm in ['kt', 'og']:
        train_farm = 'poultry'
    elif farm == 'mn':
        train_farm = 'dairy'
    split_set = args.split_set
    
    # Get features
    print('Getting features...')
    train = pd.read_csv(f"features/{train_farm}_train.csv")
    val = pd.read_csv(f"features/{farm}_{split_set}.csv")
    # resnet columns are all the columns that are named with numbers
    resnet_cols = [col for col in train.columns if col.isdigit()]

    # Get results
    print('Getting results...')
    res = get_results(farm, train, val, resnet_cols, only_preds)
    if only_preds:
        res.to_csv(f"features/preds/{farm}_{split_set}_preds.csv", index=False)
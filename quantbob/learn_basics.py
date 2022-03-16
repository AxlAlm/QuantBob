
"""

https://xgboost.readthedocs.io/en/latest/tutorials/dask.html


https://medium.com/rapids-ai/a-new-official-dask-api-for-xgboost-e8b10f3d1eb7

"""

import os
import optuna
from xgboost import XGBRegressor

from quantbob.data import NumerAIDataset

import xgboost as xgb
import dask.data as dd
from dask_cuda import LocalCUDACluster
from dask.distributed import Client



def eval_error_metric(predt, ddf: xgb.DMatrix):
    label = dtrain.get_label()
    r = np.zeros(predt.shape)
    return 'CustomErr', np.sum(r)


def early_stopping():
    """sets up early stopping"""
    return xgb.callback.EarlyStopping(
        rounds=4,
        metric_name="corr",
        data_name="val",
        save_best=False,
    )

    
def objective(trial: optuna.trial.Trial) -> float:

    # get dataset
    dataset = NumerAIDataset(debug = os.environ["debug"])
    parquet_file_paths = dataset.create_splits()
    
    # set up model and select trial hyperparamaters
    hyperparameters = {
                "tree_method": "gpu_hist", # hist
                "objective": "reg:squarederror",
                "seed": 0,
                "eta": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                "gamma": trial.suggest_int("lambda", 0, 10, log=True),
                "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True)
                }
    num_boost_round = trial.suggest_categorical("num_boost_round", [10,50,100])
        
    # set up the cluster for 
    cluster = LocalCUDACluster(n_workers=2)
    client = Client(cluster)
        
    corrs = []
    for train_fp, val_fp in parquet_file_paths:
        
        train_df = dd.read_parquet(train_fp)  
        val_df = dd.read_parquet(val_fp)  
        
        y = train_df['label']
        X = train_df.filter(regex='^feature_', axis=1)
    
    
        dtrain = xgb.dask.DaskDMatrix(client, train_df)
        dval = xgb.dask.DaskDMatrix(client, val_df)
        
        output = xgb.dask.train(
            client,
            params = hyperparameters,
            dtrain=dtrain,
            evals=[(dval, "val")],
            num_boost_round=num_boost_round,
            callbacks=[early_stopping()],
        )
        
        history = output['history']
        corrs.append(output)
    
    return stable_mean_corr(corrs)
        
    


def basic():
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    
    commet_logger.log_config(study.best_params)

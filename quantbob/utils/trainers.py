
import os

import numpy as np
import xgboost as xgb
from xgboost import dask as dxgb
from dask import dataframe as dd
from dask_cuda import LocalCUDACluster
from distributed import Client
import pytorch_lightning as pl

from quantbob.utils.datamodule import DataModule
from quantbob.utils.evaluation import spearmans

def sprearmans_wrapper(predt: np.ndarray, Xy: xgb.DMatrix):
    y = Xy.get_label()
    return ("spearman", spearmans(y, predt))

class DaskTrainer:
    """
    https://xgboost.readthedocs.io/en/latest/tutorials/dask.html

    https://medium.com/rapids-ai/a-new-official-dask-api-for-xgboost-e8b10f3d1eb7

    https://developer.nvidia.com/blog/accelerating-xgboost-on-gpu-clusters-with-dask/

    """
    def _dask_fit(self, client:Client, datamodule:DataModule, hyperparamaters:dict) -> float:
        
        Xy = dxgb.DaskDeviceQuantileDMatrix(client, *datamodule.get_xy())
        Xy_valid = dxgb.DaskDMatrix(client, *datamodule.get_val_xy())
        
        params = {
                    "objective": "reg:squarederror",
                    "eval_metric": "spearman",
                    "tree_method": "gpu_hist",
                }
        params.update(hyperparamaters)      
        
        # set up early stopping
        es = xgb.callback.EarlyStopping(
                rounds=params.get("es_rounds", 4),
                metric_name="spearman",
                data_name="Valid",
                save_best=False,
            )
        
        # train models
        output  = xgb.dask.train(
            client,
            params,
            Xy,
            evals=[(Xy_valid, "Valid")],
            num_boost_round=params.get("num_boost_round", 100),
            feval=sprearmans_wrapper,
            callbacks=[es],
        )
        return output 
        

    def fit(self, datamodule:DataModule, hyperparamaters:dict) -> float:
        
        # set up the cluster for 
        with LocalCUDACluster(n_workers=2) as cluster:
            with Client(cluster) as client: 
                output = self._dask_fit(
                    client = client,
                    datamodule = datamodule,
                    hyperparamaters = hyperparamaters
                )

        return output


class PytorchLightningTrainer:
    
    def ___init__(self, *args, **kwargs):
        self._trainer = pl.Trainer(*args, **kwargs)   

    def fit(self, nn:pl.LightningModule, datamodule:DataModule, hyperparamaters:dict) -> float:
        model = nn(**hyperparamaters)
        self._trainer.fit(model, datamodule=datamodule)
        return self._trainer.callback_metrics["val_spearmans"].item()

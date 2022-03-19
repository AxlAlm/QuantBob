
from argparse import ArgumentParser

import torch

from quantbob.data.numerai_dataset import NumerAIDataset
from quantbob.utils.trainers import DaskTrainer
from quantbob.utils.cv_objective import CrossValidationObjective
from quantbob.models.v1 import V1


    # hyperparameters = {
    #             "tree_method": "gpu_hist", # hist
    #             "objective": "reg:squarederror",
    #             "seed": 0,
    #             "eta": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
    #             "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
    #             "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True)
    #             }
    # num_boost_round = trial.suggest_categorical("num_boost_round", [10,50,100])
    
    
def setup(
    dataset:NumerAIDataset,
    study: bool = False,
    ) -> float:
    
    # set up trainer
    trainer = DaskTrainer()
    
    if study:
        objective = CrossValidationObjective(
            datamodules = dataset.get_vc_datamodules(),
            trainer = trainer,
        )
        

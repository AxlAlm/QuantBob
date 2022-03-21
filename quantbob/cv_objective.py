
from typing import Callable, Union
import os

import optuna as opt
import numpy as np
from comet_ml import Experiment
import pytorch_lightning as pl

from quantbob.trainers import DaskTrainer, PytorchLightningTrainer
from quantbob.utils import dict2uid


def cv_pruning(trial: opt.trial.Trial, score:float, cv_i:int) -> None:
    """
    Checks the performance of the current cv and decided whether to 
    prune this cv trail. 
    
    will prune if any of the below are true:
    
        a. cv at step i is 5% worse than then best cv at i

    this is to make sure that we dont fit multiple models with same params if
    it at start already produces significantly worse models than the best params
    
    
        b. the standard deviation of cvs until i is lower than 10%
    
    this is to make sure that we param produces stable models across the csv, as this is
    very important for generalization
    """
    
    # skip first trail
    if len(trial.study.get_trials()) == 1:
        return False

    # each split at i needs to be better than
    # best trial at i
    best_values = trial.study.best_trial.intermediate_values
    is_worse = (1 - (score / best_values[cv_i]))  > 0.05
    
   
    # standard deviation needs to be higher than 0.1 
    trial_values =  list(trial.storage.get_trial(trial._trial_id).intermediate_values.values())
    std = np.std(np.array(trial_values))
    is_unstable = std > 0.1
    
    if any([is_worse, is_unstable]):
        raise opt.TrialPruned()
    
    
    
class CrossValidationObjective:
    
    
    def __init__(
        self, 
        datamodules:list, 
        get_hps:Callable, 
        trainer:Union[PytorchLightningTrainer, DaskTrainer], 
        nn:pl.LightningModule = None
        ) -> None:
        self._datamodules = datamodules
        self._trainer = trainer
        self._nn = nn
        self._get_hps = get_hps
    
    
    def __call__(self, trial: opt.trial.Trial) -> float:
        
        # log hyperparameters
        hyperparamaters = self._get_hps(trial)       
                
        experiment = Experiment(
            api_key=os.environ.get("COMET_API_KEY"),
            project_name="quantbob",
            experiment_key=dict2uid(hyperparamaters)
        )
    
        # log hyperparameters
        experiment.log_parameters(hyperparamaters)
        
        # get some tags:
        experiment.add_tags([
            str(self._nn) if isinstance(self._trainer, PytorchLightningTrainer) else "xgboost",
        ])
        
        
        scores = []
        for i, datamodule in enumerate(self._datamodules):
            
            # fit model
            score = self._trainer.fit(datamodule, hyperparamaters)
            
            # store score
            scores.append(score)
            
            # report the score as a intermediate value
            trial.report(score, i)
            
            # log metric
            experiment.log_metrics({"spearmans": score})
            
            cv_pruning(trial, score, i)
            
        
        return np.mean(scores)




        
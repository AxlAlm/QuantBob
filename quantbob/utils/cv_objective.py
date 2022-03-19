
from typing import Union

import optuna as opt
import numpy as np
from comet_ml import Experiment
import pytorch_lightning as pl

from quantbob.utils.trainers import DaskTrainer, PytorchLightningTrainer
from quantbob.utils.utils import dict2uid

def select_hyperparamaters(trial:opt.trial.Trial, search_space:dict) -> dict:
    hyperparamaters = {}
    
    for k,values in search_space:
        
        hp_type = get_type(values)
    
        if hp_type == float:
            hyperparamaters[k] = trial.suggest_float(k, *values)
            
        elif hp_type == int:
            hyperparamaters[k] = trial.suggest_int(k, *values)
            
        else:
            hyperparamaters[k] = trial.suggest_categorical(k, values)
            
            
    return hyperparamaters



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
        search_space:dict, 
        trainer:Union[PytorchLightningTrainer, DaskTrainer], 
        nn:pl.LightningModule = None
        ) -> None:
        self._datamodules = datamodules
        self._trainer = trainer
        self._nn = nn
        self._search_space = search_space
    
    
    def __call__(self, trial: opt.trial.Trial) -> float:
        
  
        # log hyperparameters
        hyperparamaters = select_hyperparamaters(trial, self._search_space)       
                
        experiment = Experiment(
            api_key=os.environ.get("COMET_API_KEY"),
            project_name="QuantBob",
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
            score = self._trainer.fit(datamodule)
            
            # store score
            scores.append(score)
            
            # report the score as a intermediate value
            trial.report(score, i)
            
            # log metric
            experiment.log_metrics({"spearmans": score})
            
            cv_pruning(trial, score, i)
            
        
        return np.mean(scores)




        
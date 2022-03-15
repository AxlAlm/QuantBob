

# 
from typing import List, Tuple, Callable

# sklearn
from sklearn import pipeline
from sklearn.base import BaseEstimator

# optuna
import optuna as ot

# quantbob
from .preprocessors import get_preprocessor
from .regressors import get_regressor



def make_pipeline_maker(preprocessor_setups: List[Tuple[BaseEstimator, dict]]) -> Callable:
    
    def make_pipeline():
        return pipeline.make_pipeline(*[obj(**hps) for obj, hps in preprocessor_setups])

    return make_pipeline


def make_regressor_maker(regressor_obj, hyperparamaters) -> Callable:
    
    def make_regressor():
        return regressor_obj(**hyperparamaters)

    return make_regressor


def get_type(item):
    
    if not isinstance(item, list):
        return get_type([item])
        
    return type(item[0])


def select_hyperparamaters(trial:ot.Trial, search_space:dict) -> dict:
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


def setup_preprocessors(trial:ot.Trial, config:dict) -> Callable:
    
    # this contains uninstantiated preprocessors with their kwargs
    # this is because in later steps we want to do Cross Validation, but
    # not change hyperparamaters over splits, hence we will re-instantiate
    # the preprocessor pipeline for each cv, not use the name.
    preprocessor_setups : List[Tuple[BaseEstimator, dict]] = []
    
    for preprocessor_name in config["preprocessors"]:
        preprocessor_obj = get_preprocessor(preprocessor_name)
        hyperparamaters_options = config[preprocessor_name]
        
        hyperparamaters = select_hyperparamaters(trial, hyperparamaters_options)
    
        preprocessor_setups.append((preprocessor_obj, hyperparamaters))
    
    return make_pipeline_maker(preprocessor_setups)



def setup_regressor(trial:ot.Trial, config:dict) -> int:
    
    # get the regressor obj
    regressor_obj = get_regressor(config["algo"])

    # select hyperparamaters
    hyperparamaters = select_hyperparamaters(trial, config[config["algo"]])
    
    
    return make_regressor_maker(regressor_obj, hyperparamaters)   
    
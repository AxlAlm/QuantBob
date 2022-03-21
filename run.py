
import os
from argparse import ArgumentParser

import torch
import optuna as opt

from quantbob.data.numerai_dataset import NumerAIDataset
from quantbob.trainers import PytorchLightningTrainer, DaskTrainer
from quantbob.cv_objective import CrossValidationObjective
from quantbob.models.v1 import V1



def xgboost_params(trial:opt.trial.Trial) -> dict:
    hyperparameters = {
                "tree_method": "gpu_hist", # hist
                "objective": "reg:squarederror",
                "seed": 0,
                "eta": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True)
                }
    num_boost_round = trial.suggest_categorical("num_boost_round", [10,50,100])
    return hyperparameters


def get_trainer(nn):
    
    if nn == "xgboost":
        # set up trainer
        trainer = DaskTrainer()
    else:
        # set up trainer
        trainer = PytorchLightningTrainer(
            enable_checkpointing=False,
            max_epochs=100,
            gpus=-1 if torch.cuda.is_available() else None,
            accelerator="ddp_cpu" if not torch.cuda.is_available() else None,
            num_processes=os.cpu_count() if not torch.cuda.is_available() else None,
            callbacks=[
                EarlyStopping(monitor="val_corr", min_delta=0.00, patience=3, verbose=False, mode="max"),
            ],
        )
    
    return trainer


def run_study(n_trials:int, nn:str=None)  -> CrossValidationObjective:
    
    kwargs = dict(
        datamodules = dataset.get_cv_datamodules(n_folds=5),
    )
    
    if nn:
        kwargs["trainer"] = get_trainer()
        kwargs["get_hps"] = nn.optuna_hyperparam_selection
        kwargs["nn"] = nn
    else:
        kwargs["trainer"] = DaskTrainer()
        kwargs["get_hps"] = xgboost_params
        
    study = opt.create_study()
    study.optimize(CrossValidationObjective(**kwargs), n_trials=n_trials)
    


def run_best():
    pass


def upload():
    pass


if __name__ == '__main__':
    
    # Initialparse some args
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-t", "--n_trials", type=int, default=20)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--study", default=False, action="store_true")
    parser.add_argument("--run_best", default=False, action="store_true")
    parser.add_argument("--upload", default=False, action="store_true")
    args = parser.parse_args()

    # set up dataset
    dataset =  NumerAIDataset(debug = args.debug)
    
    # set the model/nn if used
    nn = None
    if args.model == "V1":
        nn = V1
        
    # run hyperparamater tuning
    if args.study:
        run_study(nn=nn, n_trials=1 if args.debug else args.n_trials)

    # train  model with best paramaters
    if args.run_best:
        run_best()
        
    # upload the best model, tournament predictions
    if args.upload:
        upload()
        
        
    
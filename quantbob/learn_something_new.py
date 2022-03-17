



import os

import torch
import optuna
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from quantbob.data import NumerAIDataset


def fit(model, datamodule) -> float:
    trainer = pl.Trainer(
        logger=commet_logger,
        enable_checkpointing=False,
        max_epochs=100,
        gpus=-1 if torch.cuda.is_available() else None,
        accelerator="ddp_cpu" if not torch.cuda.is_available() else None,
        num_processes=os.cpu_count() if not torch.cuda.is_available() else None,
        callbacks=[
            EarlyStopping(monitor="val_corr", min_delta=0.00, patience=3, verbose=False, mode="max"),
        ],
    )
    trainer.fit(model, datamodule=datamodule)
    return trainer.callback_metrics["val_corr"].item()


def cv_objective(trial: optuna.trial.Trial) -> float:
        
        # get dataset
        dataset = NumerAIDataset(debug = os.environ["debug"])
        
        # set up model and select trial hyperparamaters
        model, hyperparameters = V1.from_optuna(trial)
        
        # log hyperparameters
        commet_logger.log_hyperparams(hyperparameters)
        
        val_corrs = []
        for i, datamodule in enumerate(dataset.cv_splits()):
            
            # train model for one split
            val_corr = fit(
                model = model,
                hyperparameters = hyperparameters
            )
            
            # store score
            val_corrs.append(val_corr)
            
            # report the score as a intermediate value
            trial.report(val_corr, i)

            # REMAKE SO THAT WE HAVE A CV PRUNER!
            # IT SHOULD STOP THE CV IF:
            # - The inital CV is worse than the inital cv of best trail
            # - if the mean scores of CVS is lower then the best, we stop the CV
            if trial.should_prune():
                raise optuna.TrialPruned()
                
        return stable_mean_corr(val_corrs, std_threshold = 0.05)
        
    


def tune():
    study = optuna.create_study(
        study_name="pl_ddp",
        direction="maximize",
        pruner=optuna.pruners.MedianPruner() 
    )
    
    study.optimize(objective, n_trials=100, timeout=600)
    trial = study.best_trial

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
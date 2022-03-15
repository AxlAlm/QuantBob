
import os

import torch
import optuna
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from quantbob.data import NumerAIDataset

def objective(trial: optuna.trial.Trial) -> float:

    dataset = NumerAIDataset(debug = self._debug)
    model, hyperparameters = V1.from_optuna(trial)
    
    corrs = []
    for datamodule in dataset.cv_splits():
        
        trainer = pl.Trainer(
            logger=True,
            enable_checkpointing=False,
            max_epochs=100,
            gpus=-1 if torch.cuda.is_available() else None,
            accelerator="ddp_cpu" if not torch.cuda.is_available() else None,
            num_processes=os.cpu_count() if not torch.cuda.is_available() else None,
            callbacks=[
                EarlyStopping(monitor="val_corr", min_delta=0.00, patience=3, verbose=False, mode="max"),
                PyTorchLightningPruningCallback(trial, monitor="val_corr")
            ],
        )
        
        trainer.logger.log_hyperparams(hyperparameters)
        trainer.fit(model, datamodule=datamodule)
        corrs.append(trainer.callback_metrics["val_corr"].item())
    
    return stable_mean_corr(corrs)
        
    


def tune():
    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    )

    storage = "sqlite:///example.db"
    study = optuna.create_study(
        study_name="pl_ddp",
        storage=storage,
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
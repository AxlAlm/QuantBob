
from argparse import ArgumentParser

import torch

from quantbob.data.numerai_dataset import NumerAIDataset
from quantbob.utils.trainers import PytorchLightningTrainer
from quantbob.models.v1 import V1


def setup(
    commet_logger: None,
    datatset:NumerAIDataset,
    study: bool = False,
    
):

    # set up trainer
    trainer = PytorchLightningTrainer(
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
    
    
    if study:
        search_space = V1.get_optuna_search_space()
        objective = CrossValidationObjective(
            logger = commet_logger,
            datamodules = dataset.get_vc_datamodules(),
            trainer = trainer,
            nn = V1
        )

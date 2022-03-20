
from argparse import ArgumentParser
import os

import torch

from quantbob.data.numerai_dataset import NumerAIDataset
from quantbob.utils.trainers import PytorchLightningTrainer, DaskTrainer
from quantbob.utils.cv_objective CrossValidationObjective
from quantbob.models.v1 import V1



def get_trainer():
    
    if model == "xgboost":
        # set up trainer
        trainer = DaskTrainer()
        
    else:
        # set up trainer
        trainer = PytorchLightningTrainer(
            logger=comet_logger,
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



def setup_study():
    
    kwargs = dict(
        datamodules = dataset.get_vc_datamodules(),
    )

    kwargs["search_space"] = get_trainer()
    
    if nn != "xgboost":
        kwargs["search_space"] = V1.get_optuna_search_space()
        kwargs["nn"] = get_nn(nn)


    return CrossValidationObjective(**kwargs)
    

def setup_best_run(
    comet_logger: None,
    dataset:NumerAIDataset,
    study: bool = False,
    
    ):
    pass



def upload_best(
    comet_logger: None,
    dataset:NumerAIDataset,
    study: bool = False,
    
    ):
    pass




if __name__ == '__main__':
    
    # Initialparse some args
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--study", default=False, action="store_true")
    parser.add_argument("--run_best", default=False, action="store_true")
    parser.add_argument("--upload", default=False, action="store_true")
    args = parser.parse_args()

    # set up dataset
    os.enviro["debug"] = args.debug
    dataset =  NumerAIDataset()

    # run hyperparamater tuning
    if args.tune:
        run_study()

    # train  model with best paramaters
    if args.run_best:
        run_best()
        
    # upload the best model, tournament predictions
    if args.upload:
        upload()
        
        
    
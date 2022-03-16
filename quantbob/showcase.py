

    
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping



def bob():
    
    # arguments made to CometLogger are passed on to the comet_ml.Experiment class
    comet_logger = pl.loggers.CometLogger(**commet_logger_config)
    
    
    comet_logger = CometLogger(
        api_key=os.environ.get("COMET_API_KEY"),
        workspace=os.environ.get("COMET_WORKSPACE"),  # Optional
        save_dir=".",  # Optional
        project_name="default_project",  # Optional
        rest_api_key=os.environ.get("COMET_REST_API_KEY"),  # Optional
        experiment_key=os.environ.get("COMET_EXPERIMENT_KEY"),  # Optional
        experiment_name="lightning_logs",  # Optional
    )
        
    
    # set up dataset
    dataset = NumerAIDataset(debug = self._debug)
    
    # set up model
    model, hyperparameters = V1(BEST_HYPERPARAMATERS)
    
    # set up trainger
    trainer = pl.Trainer(
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
    
    # train model
    trainer.fit(model, datamodule=dataset.train_datamodule())


    # test model
    test_results = trainer.test(model, datamodule=dataset.validation_datamodule())
    comet_logger.logger(test_results)

    # predict
    predictions = trainer.predict(model, datamodule=dataset.tournament_datamodule())

    # upload predictions

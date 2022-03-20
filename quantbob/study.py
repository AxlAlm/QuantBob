
from argparse import ArgumentParser

from quantbob.utils.trainers import PytorchLightningTrainer
from quantbob.models.v1 import V1


def setup_pl():

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
    
    # 
    search_space = V1.get_optuna_search_space()

    objective = CrossValidationObjective(
        logger = commet_logger,
        datamodules = datamodules,
        trainer = trainer,
        nn = V1
    )


def setup_dask():
    pass


if __name__ == '__main__':
    
    # Initialparse some args
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--tune", default=False, action="store_true")
    args = parser.parse_args()

    
    # set up dataset
    datamodules =  NumerAIDataset(debug = self._debug).get_cv_datamodules()

   
  
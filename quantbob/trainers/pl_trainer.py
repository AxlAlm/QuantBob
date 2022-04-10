import pytorch_lightning as pl

from quantbob.data.datamodule import DataModule


class PytorchLightningTrainer:
    def ___init__(self, *args, **kwargs):
        self._trainer = pl.Trainer(*args, **kwargs)

    def _parse_args(*args, **kwargs):
        pass
        # enable_checkpointing=False,
        # max_epochs=100,
        # gpus=-1 if torch.cuda.is_available() else None,
        # accelerator="ddp_cpu" if not torch.cuda.is_available() else None,
        # num_processes=os.cpu_count() if not torch.cuda.is_available() else None,
        # callbacks=[
        #     EarlyStopping(
        #         monitor="val_corr",
        #         min_delta=0.00,
        #         patience=3,
        #         verbose=False,
        #         mode="max",
        #     ),
        # ],

    def fit(
        self, model: pl.LightningModule, datamodule: DataModule, hyperparamaters: dict
    ) -> float:
        model = model(**hyperparamaters)
        self._trainer.fit(model, datamodule=datamodule)
        return self._trainer.callback_metrics["val_spearmans"].item()

from quantbob.trainers.dask_trainer import DaskTrainer


class QuantBob:
    def __init__(
        self,
        model: Union[str, pl.LightningModule],
        training_params: dict,
        debug: bool = False,
    ):
        self._debug = debug
        self._model = model
        self._training_params = training_params
        self._dataset = NumerAIDataset(debug=self._debug)

    def _setup_trainer(self) -> Union[DaskTrainer, PytorchLightningTrainer]:

        if self._model.__name__ == "xgboost":
            # set up trainer
            trainer = DaskTrainer(**training_params)
        else:
            # set up trainer
            trainer = PytorchLightningTrainer(
                enable_checkpointing=False,
                max_epochs=100,
                gpus=-1 if torch.cuda.is_available() else None,
                accelerator="ddp_cpu" if not torch.cuda.is_available() else None,
                num_processes=os.cpu_count() if not torch.cuda.is_available() else None,
                callbacks=[
                    EarlyStopping(
                        monitor="val_corr",
                        min_delta=0.00,
                        patience=3,
                        verbose=False,
                        mode="max",
                    ),
                ],
            )

    def study(self, n_folds: int, n_trials: int) -> None:

        cv_obj = CrossValidationObjective(
            model=self._model,
            datamodules=self._dataset.get_cv_datamodules(n_folds=n_folds),
            trainer=self._setup_trainer(),
        )

        study = opt.create_study()
        study.optimize(cv_obj, n_trials=n_trials)

    def report():
        pass

    def create():
        pass

    def upload():
        pass

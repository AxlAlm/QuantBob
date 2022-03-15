



def objective(trial: optuna.trial.Trial, dataset) -> float:


    corrs = []
    for datamodule in dataset:
        
        model, hyperparameters = V1.from_optuna(trial)

        trainer = pl.Trainer(
            logger=True,
            enable_checkpointing=False,
            max_epochs=EPOCHS,
            gpus=-1 if torch.cuda.is_available() else None,
            accelerator="ddp_cpu" if not torch.cuda.is_available() else None,
            num_processes=os.cpu_count() if not torch.cuda.is_available() else None,
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_corr")],
        )
        
        trainer.logger.log_hyperparams(hyperparameters)
        trainer.fit(model, datamodule=datamodule)

        corrs.append(trainer.callback_metrics["val_acc"].item())
    
    
    mean_corr = np.mean(corrs)
    std = np.std(corrs)
    
    if std > 0.0:
        pass
        
    


def run_optuna_study():
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
        
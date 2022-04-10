class Xgboost:
    """Just a helper class to make xgboost conform to the nn structure of all models"""

    @classmethod
    def optuna_hyperparam_selection(self, trial: opt.trial.Trial) -> dict:
        hyperparameters = {
            "tree_method": "gpu_hist",  # hist
            "objective": "reg:squarederror",
            "seed": 0,
            "eta": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        }
        num_boost_round = trial.suggest_categorical("num_boost_round", [10, 50, 100])
        return hyperparameters

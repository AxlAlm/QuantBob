
# basics
from typing import List, Dict
import pandas as pd
import os
import toml
import pwd
from pprint import pprint

# sklearn 
from sklearn.pipeline import Pipeline

# quantbob
from quantbob import napi
from .numerai_dataset import NumerAIDataset, ParquetCVReader
from .evaluation import correlation_score
from .optuna_setup import setup_preprocessors, setup_regressor


user_dir = pwd.getpwuid(os.getuid()).pw_dir


class QuantBob:


    def __init__(self, config_path:str, debug : bool = False, force_download : bool = False) -> None:
        
        # set the mode to debug mode and will use fraction of the training data
        self._debug = debug

        # init the dataset
        self._dataset = NumerAIDataset(debug = self._debug, force_download = force_download)

        # setup
        self._path_to_round = os.makedirs(f"{user_dir}/.quantbob/{self._dataset.current_round}", exist_ok=True)

        # read config
        config : dict = self._read_config(config_path)
        
        pprint(config)
        
        # one model with one set of hyperparamaters is one "task". We parase out
        # all these task form the config into list so we can iterate over them
        tasks : List[Dict] = self._create_tasks()
        
        
        self._run_tasks(tasks)


    def _read_config(self, config_path:str) -> dict:

        with open(config_path) as f:
            config = toml.load(f)

        return config


    def _create_tasks(self, config : dict) -> List[dict]:
        print(config)
        print(lol)


    def _run_tasks(self, tasks : List[dict]) -> None:
        
        for task in tasks:
            
            study = optuna.create_study()
            
            self._task(task = task)
            study.optimize(objective, n_trials=100)
    
    
    def _extract_xy(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = df.filter(regex='^feature_', axis=1)
        y = df.filter(regex='^target_', axis=1)
        return X, y
    
    
    def _train(self, df:pd.DataFrame, preprocessor:BaseEstimator, regressor:BaseEstimator) -> None:
        X, y = self._extract_xy(df)
        X = preprocessor.fit_transform(X)
        regressor.fit(X, y)
    
    
    def _test(self, df:pd.DataFrame, preprocessor:BaseEstimator, regressor:BaseEstimator) -> None:
        X, y = self._extract_xy(df)
        X = preprocessor.transform(X)
        pred_y = regressor.predict(X)
        corr = correlation_score(y, pred_y)
        return corr
        
    
    def _task(self, trial, task: dict) -> None:
        
        # create generator which will read the cross validations
        # parquet files
        cv_iter : ParquetCVReader = self._dataset.create_cvs()
        
        # create a list to append rows for each cv run
        split_corrs = []
        
        # select hyperparamaters via optuna for the task, returns a
        # function to generate new instances
        make_preprocessor : Callable = setup_preprocessors(trial, config)
        make_regressor : Callable = setup_regressor(trial, config)    
        
        
        for cv_id, (train_df, test_df) in enumerate(cv_iter):
            
            # make a new instance of the preprocessor
            preprocessor = make_preprocessor()
                    
            # make a new instance of the regressor
            regressor = make_regressor()
            
            self._train(df=train_df, preprocessor=preprocessor, regressor=regressor)
            corr = self._test(df=train_df, preprocessor=preprocessor, regressor=regressor)

       
            split_corrs.append(corr)
            
            
            # MAYBE / if the score is shit we can end the trail already
            # during on of the CVS?
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        
        
        mean_corr = np.mean(split_corrs)
        std = np.std(split_corrs)
        
        if std > 0.0
    



    # auxillary targets
    # if we want to train with auxillary targets 
    # we need a strategy, here are some ideas:
    # 
    # 1) train the model again with auxillary targets
    # 2) use sklearn.multioutput.RegressorChain
    #
    # maybe we dont need to set a strategy just 
    # make sure the model is a RegressorChain for example
    # if using sklearn models or, if the model is a 
    # neural network that is supports multi targets.
    # self._auxillary_targets = []
    # self._target : str = ""
    # self._targets = [self._target] + self._auxillary_targets


    # @classmethod
    # def rankings(self, round:int, top:int = 3) -> pd.DataFrame:
        
    #     # set the folder to the round
    #     path_to_round = f"~/.quantbob/{round}"

    #     # check if folder for round exists
    #     if not os.path.exists(path_to_round):
    #         raise KeyError("No data for that round")

    #     # load the rankings
    #     rank_df = pd.read_csv(os.path.join(path_to_round, "rankings.csv"), index_col=0)

    #     return rank_df
    

    # @classmethod
    # def compare(self, round:int, top:int = 3):

    #     # get top models for the current round
    #     top_models = sorted(napi.round_details(self._current_round), 
    #                         key = lambda x:x["correlation"], reverse=True)[:top]


    #     top_df = pd.DataFrame(top_models)

    #     return top_df


    # def run_from_config(self,
    #                     config_path:str, 
    #                     ) -> None:


    #     # setup cross validations
    #     cv_generator = getattr(cvs, cv_config["name"])(self._dataset, **cv_config["kwargs"])

    #     # setup feature pipeline
    #     d = []

    #     for step in  feature_preprocessing_config.values():
    #         dd = getattr(feature_preprocessing, step["name"])
    #         print(dd)
    #         dd = dd(**step.get("kwargs", {}))
    #         print(dd)


    #     feature_pipeline = Pipeline([getattr(feature_preprocessing, step["name"])(**step.get("kwargs", {})) 
    #                                 for step in feature_preprocessing_config.values()])


    #     # get model stuff
    #     model_class = regressors[model_config["name"]]
    #     hyperparamaters = model_config["kwargs"]


    #     cv_models = []
    #     scores = []

    #     # ------ CV loop -----
    #     for k, (train_X, train_y), (test_X, test_y) in cv_generator:

    #         # init model for the split
    #         model = model_class(**hyperparamaters)

    #         # ---- train model ---
    
    #         # preprocess train features
    #         train_X = feature_pipeline.fit_transform(train_X)

    #         # auxillary targets
    #         # if we want to train with auxillary targets 
    #         # we need a strategy, here are some ideas:
    #         # 
    #         # 1) train the model again with auxillary targets
    #         # 2) use sklearn.multioutput.RegressorChain
    #         #
    #         # maybe we dont need to set a strategy just 
    #         # make sure the model is a RegressorChain for example
    #         # if using sklearn models or, if the model is a 
    #         # neural network that is supports multi targets.
    #         if self._auxillary_targets and self._auxillary_strategy is not None:
    #             raise NotImplementedError
    #         else:
    #             model.fit(train_X, train_y)


    #         # ----- evalute model -----
    #         # preprocess test features
    #         test_x = feature_pipeline.transform(test_X)

    #         # predict on test data
    #         test_y_pred = model.predict(test_X)

    #         # get evaluation scores
    #         model_score = evaluate(test_y_pred, test_y)

    #         scores.append(model_score)
    #         cv_models.append(model)


    #         if debug and k == 3:
    #             break


    #     # select model or 
    #     self._model_selection(cv_models, scores)


    def evaluate(self):
        pass
    

    def predict(self, tournament=True, validation=True):
        pass


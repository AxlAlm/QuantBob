
# basics
import pandas as pd
import numpy as np
import os
import yaml
import pwd
from glob import glob

# sklearn 
from sklearn.pipeline import Pipeline

# quantbob
from quantbob import napi
from .dataset import NumerAIDataset
from quantbob import feature_preprocessing
import quantbob.cv as cvs



user_dir = pwd.getpwuid(os.getuid()).pw_dir


class QuantBob:


    def __init__(self):

        # init the dataset
        self._dataset = NumerAIDataset()

        # setup
        self._path_to_round = os.makedirs(f"{user_dir}/.quantbob/{self._dataset.current_round}", exist_ok=True)

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
        self._auxillary_targets = []
        self._target : str = ""
        self._targets = [self._target] + self._auxillary_targets


    @classmethod
    def rankings(self, round:int, top:int = 3) -> pd.DataFrame:
        
        # set the folder to the round
        path_to_round = f"~/.quantbob/{round}"

        # check if folder for round exists
        if not os.path.exists(path_to_round):
            raise KeyError("No data for that round")

        # load the rankings
        rank_df = pd.read_csv(os.path.join(path_to_round, "rankings.csv"), index_col=0)

        return rank_df
    

    @classmethod
    def compare(self, round:int, top:int = 3):

        # get top models for the current round
        top_models = sorted(napi.round_details(self._current_round), 
                            key = lambda x:x["correlation"], reverse=True)[:top]


        top_df = pd.DataFrame(top_models)

        return top_df


    def __read_yml(self, file_path:str) -> dict:

        with open(file_path) as f:
            config = {f"{k}_config":v for k,v in yaml.load(f, Loader=yaml.FullLoader).items()}

        return config



    def __feature_preprocessing(self, X: np.ndarray):
        pass



    def run(self,
            feature_preprocessing_config:list, 
            cv_config:dict, 
            model_config:dict,
            aux_config:dict
            ) -> None:


        debug = True

        if debug:
            cv_config["kwargs"]["cv"] = int(self._dataset.n_train_eras / 3)


        # ----- SETUP -----
    
        # setup cross validations
        cv_generator = getattr(cvs, cv_config["name"])(self._dataset, **cv_config["kwargs"])

        # setup feature pipeline

        d = []

        for step in  feature_preprocessing_config.values():
            dd = getattr(feature_preprocessing, step["name"])
            print(dd)
            dd = dd(**step.get("kwargs", {}))
            print(dd)


        feature_pipeline = Pipeline([getattr(feature_preprocessing, step["name"])(**step.get("kwargs", {})) 
                                    for step in feature_preprocessing_config.values()])


        # get model stuff
        model_class = regressors[model_config["name"]]
        hyperparamaters = model_config["kwargs"]


        cv_models = []
        scores = []

        # ------ CV loop -----
        for k, (train_X, train_y), (test_X, test_y) in cv_generator:

            # init model for the split
            model = model_class(**hyperparamaters)

            # ---- train model ---
    
            # preprocess train features
            train_X = feature_pipeline.fit_transform(train_X)

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
            if self._auxillary_targets and self._auxillary_strategy is not None:
                raise NotImplementedError
            else:
                model.fit(train_X, train_y)


            # ----- evalute model -----
            # preprocess test features
            test_x = feature_pipeline.transform(test_X)

            # predict on test data
            test_y_pred = model.predict(test_X)

            # get evaluation scores
            model_score = evaluate(test_y_pred, test_y)

            scores.append(model_score)
            cv_models.append(model)


            if debug and k == 3:
                break


        # select model or 
        self._model_selection(cv_models, scores)


    def evaluate(self):
        pass
    

    def predict(self, tournament=True, validation=True):
        pass



    def from_yamls(self, dir_to_configs: str):
        
        for fp in glob(dir_to_configs+"/*"):
            print(fp)
            self.run(**self.__read_yml(fp))


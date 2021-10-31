
# basics
import pandas as pd
import os

# quantbob
from quantbob import napi
from .dataset import NumerAIDataset


class QuantBob:


    def __init__(self, 
                cv_method : None,
                model : None,
                feature_preprocessing: None,     
                debug_mode = True,   
                ):

        # init the dataset
        self._dataset = NumerAIDataset()

        # setup
        self._path_to_round = os.makedir(f"~/.quantbob/{self._dataset.current_round}", exist_ok=True)

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

    
    def run(self):

        
        cv_models = []
        scores = []
        for train, test in cvs:

            model = model_class(**hyperparamaters)

            # ---- train model ---
        
            # preprocess train features
            train_x = self._feature_preprocessing(train["features"])

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
                model.fit(train_x, train[self._targets])


            # ----- evalute model -----
            
            # preprocess test features
            test_x = self._feature_preprocessing(test["features"])

            # predict on test data
            test_y_pred = model.predict(test[self._targets])

            # get evaluation scores
            model_score = evalute(test_x, test[target])

            scores.append(model_score)
            cv_models.append(model)


        # select model or 
        self.model_selection(cv_models, scores)



    def evaluate(self):


        self._fit_model()
        pass
      



    def predict(self, tournament=True, validation=True):
        pass



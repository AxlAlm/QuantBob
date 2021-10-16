

# basics
import pandas as pd
import numpy as np

# sklearn
from sklearn.ensemble import BaseEnsemble


# quantbot
from .base import Model

class BasicEnsambler(Model):

    def __init__(self, model:BaseEnsemble, **kwargs) -> None:
        self._model = model(**kwargs)


    def __select_features(self, df:pd.DataFrame) -> np.ndarray:
        feature_cols = [c for c in df if c.startswith("feature_")]
        return df[feature_cols].to_numpy()


    def fit(self, df : pd.DataFrame):

        #select data
        X = self.__select_features(df)

        # set targets
        y = df["target"].to_numpy()

        #fit mdoel
        self._model.fit(X, y)


    def predict(self, df : pd.DataFrame):

        # select data
        X = self.__select_features(df)
        
        #predict
        return self._model.predict(X)

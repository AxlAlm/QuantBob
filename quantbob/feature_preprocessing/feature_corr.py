
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CorrelatedFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, threshold : float =  0.5):
        self._threshold = threshold
        self._features = None


    def fit(self, X, y = None):
        corr_matrix = np.abs(np.corrcoef(X, rowvar=False))
        self._features = [f for f in range(len(corr_matrix)) if sum(corr_matrix[f] > self._threshold) > 1]

    def transform(self, X, y = None):
        return X[:, self._features]

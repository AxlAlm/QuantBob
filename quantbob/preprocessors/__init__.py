

# sklearn
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import feature_selection
from sklearn.base import BaseEstimator

#quantbob
from .feature_corr import CorrelatedFeatures

def get_preprocessor(name:str) -> BaseEstimator:
    """TODO: Find a nicer way to do this?"""


    if name == CorrelatedFeatures:
        return CorrelatedFeatures
    
    
    try:
        return getattr(decomposition, name)
    except:
        pass
    
    try:
        return getattr(preprocessing, name)
    except:
        pass
    
    try:
        return getattr(feature_selection, name)
    except:
        pass
    
    raise AttributeError(f"'{name}' is not a supported preprocessor. Preprocessors are selected from preprocessors, \
                         or from sklearn.decomposition, sklearn.preprocessing, sklearn.feature_selection")
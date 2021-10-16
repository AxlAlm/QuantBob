

#basics
from abc import ABC, abstractclassmethod




class Model(ABC):

    """
    All models need to implement a fit and predict function.

    All other techniques used to make a model, e.g. feature selection, normalisation, 
    selection of ensamble splits should also be present in the model if one choses to use 
    such techniques.
    """

    # @abstractclassmethod
    # def __repr_(self):
    #     pass


    @abstractclassmethod
    def fit(self):
        pass


    @abstractclassmethod
    def predict(self):
        pass

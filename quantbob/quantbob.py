

# basics
from copy import deepcopy

# quantbot
from .dataset import Dataset
from .evaluator import Evaluator
from .data_samplers.base import DataSampler
from .model import Model

class QuantBob:


    def __init__(
                self,
                dataset : Dataset,
                model : Model,
                data_sampler : DataSampler,
                evaluator : Evaluator
                ) -> None:
        self._dataset = dataset 
        self._model = model
        self._data_sampler = data_sampler
        self._evaluator = evaluator

        self._models = []


    def train(self):

        splits = self._data_sampler.sample(self._dataset)

        for split in splits:

            model = deepcopy(self._model)

            # fit model
            model.fit(split.train_x, split.train_y)

            # get validation scores
            model.val(split.val_x, split.val_y)

        
            self._models.append(model)
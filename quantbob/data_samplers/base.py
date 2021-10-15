


from abc import ABC, abstractclassmethod

class BaseDataSampler(ABC):

    @abstractclassmethod
    def sample(self):
        pass
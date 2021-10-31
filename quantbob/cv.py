

# basics
from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd


# sklearn
from sklearn.model_selection import train_test_split


## CV SPLITTING METHODS

"""
Contains different methods for doing Cross Validation of time series data

Some implementations idea/solutions are from the forums:
https://forum.numer.ai/t/era-wise-time-series-cross-validation/791

"""


class CV(ABC):


    def __init__(self, df:pd.DataFrame, cv:int, remove_leakage=True) -> None:

        # set eras as columns
        self.df = df
        self.df["era"] = self.df["era"].astype(int)
        self.df = self.df.set_index("era")

        # get all eras. If remove_leakage == True, we remove overlapping eras, i.e. we take every 5th
        self.eras = [i for i in range(1, self.df.index.nunique(), remove_leakage if True else 1)]

        # calculate split sizes
        self.split_size  = len(self.eras) // 3 
        self.test_size = self.split_size // 3

        # get list of columns
        self.feature_columns = [c for c in self.df.columns if "feature_" in c]
        self.target_columns = [c for c in self.df.columns if "target_" in c]

        # number of cross validations
        self.cv = cv


    def train_test_split(self, split_eras:list) -> Tuple[list, list]:
        return split_eras[:-self.test_size], split_eras[-self.test_size:]


    def get_xy(self, eras) -> dict:
        return {
                "features": self.df.loc[eras, self.feature_columns],
                "targets": self.df.loc[eras, self.target_columns]
                }
        
    @abstractmethod
    def __iter__(self):
        pass




class ExpandingTimeCV(CV):
    """
    
    Train and Test splits are contigious in time and training splits increase in size for each 
    time cv


    Example, if cv = 3, (X = train, Y = test):

    0. XXXXX YYYY
    1. XXXXXXXXXX YYYY
    2. XXXXXXXXXXXXXXX YYYY
    --- time ---> 
    
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


    def __iter__(self):
 
        exp_split_size = self.split_size
        for k in range(self.cv):
            
            # the the eras for the split
            split_eras = self.eras[:exp_split_size]

            # get train and test eras
            train_eras, test_eras = self.train_test_split(split_eras)
  
            # set the new split start
            self.exp_split_size += self.split_size

            # return k, and xys
            yield k, self.get_xy(train_eras), self.get_xy(test_eras)

        

class SlidingTimeCV(CV):

    """
    
    Train and Test splits are contigious in time and both test and move forward in time for each
    split.

    Example, if cv = 3, (X = train, Y = test, - = unused dat):

    0. XXXXX YYYY
    1. -- XXXXX YYYY
    2. ------ XXXXXX YYYY
    --- time ---> 
    
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        
    def __iter__(self):
 
        split_start = 0
        for k in range(self.cv):
            
            # the the eras for the split
            split_eras = self.eras[split_start:split_start + self.split_size]

            # get train and test eras
            train_eras, test_eras = self.train_test_split(split_eras)
  
            # set the new split start
            split_start += self.split_size
            
            # return k, and xys
            yield k, self.get_xy(train_eras), self.get_xy(test_eras)

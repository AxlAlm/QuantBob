
#basics
import os
import pandas as pd
from pathlib import Path
import pytz
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

#h5py
#mport h5py

# numerapi
from numerapi import NumerAPI

# halo
from halo import Halo


napi = NumerAPI()



class Dataset:

    """
    Loads the numerai dataset
    
    """

    def __init__(self, force_download :bool =  False) -> None:

        ## setup files
        self._path_to_data = "/tmp/numerai_data"
        os.makedirs(self._path_to_data, exist_ok = True)
        self._train_fp : str = os.path.join(self._path_to_data, "numerai_training_data.parquet")
        self._val_fp : str = os.path.join(self._path_to_data, "numerai_validation_data.parquet")
        self._tournament_fp : str  = os.path.join(self._path_to_data,"numerai_tournament_data.parquet")
        self._last_updated : str = os.path.join(self._path_to_data,"last_updated.txt")

        #download data
        self.__download_data(force_download = force_download)


    @property
    def train_data(self):
        return self.__load_data("train")


    @property
    def val_data(self):
        return self.__load_data("val")


    @property
    def tournament_data(self):
        return self.__load_data("tournament")


    def __load_data(self, split:str):
        spinner = Halo(text=f'Loading {split} data', spinner='dots')
        spinner.start(f'Loading {split} data')
        df = pd.read_parquet(getattr(self, f"_{split}_fp"))
        spinner.succeed()
        return df


    def __check_new(self) -> bool:
        """
        Will check if there has been an update to numerai data within 24 hours and if the data 
        we have download have been downloaded before that, if so we return True, else False
        
        """

        # set the timezone
        sweden = pytz.timezone('Europe/Stockholm')
        current_time = datetime.now().astimezone(sweden)


        # if we dont have a file for last update we create one and return true
        if not os.path.exists(self._last_updated):

            with open(self._last_updated, "w") as f:
                f.write(str(current_time.timestamp()))

            return True


        # if we have a last update file, we take the timestamp and make it into a datetime object
        with open(self._last_updated, "r") as f:
            last_time = datetime.fromtimestamp(float(f.read().strip()), tz = sweden)


        # then we compare the number of hours since the last update
        time_diff = current_time - last_time
        over_24h = time_diff.total_seconds() / 3600 > 24

        # check if there is a new round started within 24 hours
        new_data_last_24h = napi.check_new_round()


        # if we have new data within the last 24 hours and our data was last
        #  updated in longer than 24 hours we will return true
        return over_24h and new_data_last_24h


    def __check_exist(self) -> bool:
        # check if the files exist
        no_data = not (
            os.path.exists(self._train_fp)
            and os.path.exists(self._val_fp)
            and  os.path.exists(self._tournament_fp)
        )


    def __download_data(self, force_download :bool) -> None:

        """
        downloading the current data using the numerapi

        from https://github.com/numerai/example-scripts/blob/master/example_model.py
        
        """

        # check if there is new data
        if self.__check_exist() or self.__check_new() or force_download:

            # read in all of the new datas
            # tournament data and example predictions change every week so we specify the round in their names
            # training and validation data only change periodically, so no need to download them over again every single week
            napi.download_dataset("numerai_training_data.parquet", self._train_fp)
            napi.download_dataset("numerai_validation_data.parquet", self._val_fp)
            napi.download_dataset("numerai_tournament_data.parquet", self._tournament_fp)

            #napi.download_dataset("example_predictions.parquet", os.path.join(path_to_data, "example_predictions.parquet"))
            #napi.download_dataset("example_validation_predictions.parquet", os.path.join(path_to_data, "example_validation_predictions.parquet"))


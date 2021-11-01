
#basics
import os
import pandas as pd
from pathlib import Path
import pytz
from datetime import datetime
from typing import Optional
import json

# halo
from halo import Halo

# napi
from quantbob import napi


class NumerAIDataset:

    """
    Loads the numerai dataset
    
    """

    def __init__(self, force_download :bool =  False) -> None:

        ## setup directory
        self._path_to_data = "/tmp/numerai_data"
        os.makedirs(self._path_to_data, exist_ok = True)

        # prev current_round file
        self._info_fp : str = os.path.join(self._path_to_data, "info.json")
        self._info = self.__load_info()

        # set datafiles
        self._train_fp : str = os.path.join(self._path_to_data, "numerai_training_data.parquet")
        self._val_fp : str = os.path.join(self._path_to_data, "numerai_validation_data.parquet")
        self._tournament_fp : str  = os.path.join(self._path_to_data, "numerai_tournament_data.parquet")
        self._train_h5_fp : str  = os.path.join(self._path_to_data, "train_data.h5")

        # set current round
        self._current_round = napi.get_current_round()


        # check if there is new data
        if self.__check_data_exist() or self.__check_new_round_exist() or force_download:

            # download data
            self.__download_data(force_download = force_download)

            # transform the training data into h5 so we can 
            # index per era quickly without haveing the whole dataset 
            # in memory
            self.__create_train_hdf()


            ## info
            self.__update_info()


        # extract the names for features and columns
        self._feaure_cols , self._target_cols = self.__extract_column_value()

    @property
    def train_data(self):
        return self.__load_data("train")


    @property
    def val_data(self):
        return self.__load_data("val")


    @property
    def tournament_data(self):
        return self.__load_data("tournament")


    @property
    def current_round(self):
        return self._current_round("train")


    @property
    def features(self):
        return self._feaure_cols


    @property
    def targets(self):
        return self._target_cols


    @property
    def n_train_eras(self):
        return self._n_train_eras


    def __load_info(self) -> dict:

        if not os.path.exists(self._info_fp):
            return {}

        with open(self._info_fp, "r") as f:
            info = json.load(f)

        return info


    def __update_info(self) -> None:
        with open(self._info_fp, "w") as f:
            json.dump(self._info, f)


    def get_train_xy(self, eras:list) -> pd.DataFrame:

        with pd.HDFStore(self._train_h5_fp, mode = "r") as h5_storage:
            df = h5_storage.select("df", where=f'index in {eras}')

        return df[self.features].to_numpy(), df[self.targets].to_numpy()


    def __load_prev_round(self) -> int:
    
        # if we dont have a file for last update we create one and return true
        if not os.path.exists(self._prev_round_fp):
            return None

        with open(self._prev_round_fp, "r") as f:
            prev_round = int(f.read().strip())

        return prev_round


    def __update_prev_round(self) -> int:
    
        with open(self._prev_round_fp, "w") as f:
            f.write(str(self._current_round))


    def __load_data(self, split:str):
        spinner = Halo(text=f'Loading {split} data', spinner='dots')
        spinner.start(f'Loading {split} data')
        df = pd.read_parquet(getattr(self, f"_{split}_fp"))
        spinner.succeed()
        return df


    def __check_new_round_exist(self) -> bool:

        prev_round = self.__load_prev_round()
        new_round_exists = prev_round != self._current_round

        if new_round_exists:
            self.__update_prev_round()
        
        return new_round_exists


    def __check_data_exist(self) -> bool:
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
        # read in all of the new datas
        # tournament data and example predictions change every week so we specify the round in their names
        # training and validation data only change periodically, so no need to download them over again every single week
        napi.download_dataset("numerai_training_data.parquet", self._train_fp)
        napi.download_dataset("numerai_validation_data.parquet", self._val_fp)
        napi.download_dataset("numerai_tournament_data.parquet", self._tournament_fp)
        #napi.download_dataset("example_predictions.parquet", os.path.join(path_to_data, "example_predictions.parquet"))
        #napi.download_dataset("example_validation_predictions.parquet", os.path.join(path_to_data, "example_validation_predictions.parquet"))

    
    def __create_train_hdf(self) -> None:

        if os.path.exists(self._train_h5_fp):
            return 
        
        # load in all the training data
        df = self.train_data()

        # set index to eras
        df["era"] = df["era"].astype(int)
        df = df.set_index("era")

        # create hdf file
        df.to_hdf(self._train_h5_fp, key='df', mode='w', format='table')


    def __extract_column_value(self):
        df = self.get_train_eras(eras = [1])
        return ([c for c in df.columns if "feature_" in c], 
                [c for c in df.columns if "target_" in c])


   # def __check_new(self) -> bool:
    #     """
    #     Will check if there has been an update to numerai data within 24 hours and if the data 
    #     we have download have been downloaded before that, if so we return True, else False
        
    #     """

    #     # set the timezone
    #     sweden = pytz.timezone('Europe/Stockholm')
    #     current_time = datetime.now().astimezone(sweden)


    #     # if we dont have a file for last update we create one and return true
    #     if not os.path.exists(self._last_updated):

    #         with open(self._last_updated, "w") as f:
    #             f.write(str(current_time.timestamp()))

    #         return True


    #     # if we have a last update file, we take the timestamp and make it into a datetime object
    #     with open(self._last_updated, "r") as f:
    #         last_time = datetime.fromtimestamp(float(f.read().strip()), tz = sweden)


    #     # then we compare the number of hours since the last update
    #     time_diff = current_time - last_time
    #     over_24h = time_diff.total_seconds() / 3600 > 24

    #     # check if there is a new round started within 24 hours
    #     new_data_last_24h = napi.check_new_round()


    #     # if we have new data within the last 24 hours and our data was last
    #     #  updated in longer than 24 hours we will return true
    #     return over_24h and new_data_last_24h
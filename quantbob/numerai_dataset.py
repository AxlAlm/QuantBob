
#basics
from typing import Tuple
import os
import pandas as pd
import numpy as np
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

    def __init__(self, debug:bool = False, force_download :bool =  False) -> None:

        # _debug
        self._debug = debug

        ## setup directory
        self._path_to_data = "/tmp/numerai_data"
        os.makedirs(self._path_to_data, exist_ok = True)

        # prev current_round file
        self._info_fp : str = os.path.join(self._path_to_data, "info.json")
        self._info = self._load_info()

        # set datafiles
        self._train_fp : str = os.path.join(self._path_to_data, "numerai_training_data.parquet")
        self._val_fp : str = os.path.join(self._path_to_data, "numerai_validation_data.parquet")
        self._tournament_fp : str  = os.path.join(self._path_to_data, "numerai_tournament_data.parquet")
        self._train_h5_fp : str  = os.path.join(self._path_to_data, "train_data.h5")
        self._sample_h5_fp : str = os.path.join(self._path_to_data, "sample_data.h5")

        # set current round
        self._current_round = napi.get_current_round()


        # check if there is new data
        if not self._data_exist() or self._new_round_exist() or force_download:

            # download data
            self._download_data(force_download = force_download)


        if not os.path.exists(self._train_h5_fp):

            # transform the training data into h5 so we can 
            # index per era quickly without haveing the whole dataset 
            # in memory
            self._create_train_hdf()


            # creates a small sample dataset from train to be used for debug
            if not os.path.exists(self._sample_data):
                self._create_sample_data()

            ## info
            self._update_info()

 
        # extract the names for features and columns
        self._feaure_cols , self._target_cols = self._get_column_values()


    @property
    def train_data(self):
        return self._load_data("train")


    @property
    def val_data(self):
        return self._load_data("val")


    @property
    def tournament_data(self):
        return self._load_data("tournament")


    @property
    def current_round(self):
        return self._current_round


    @property
    def features(self):
        return self._feaure_cols


    @property
    def targets(self):
        return self._target_cols


    @property
    def n_train_eras(self):
        return self._info["n_train_eras" if not self._debug else "n_sample_eras"]


    def _load_info(self) -> dict:

        if not os.path.exists(self._info_fp):
            return {}

        with open(self._info_fp, "r") as f:
            info = json.load(f)

        return info


    def _update_info(self) -> None:
        with open(self._info_fp, "w") as f:
            json.dump(self._info, f)


    def _get_train_eras(self, where:str) -> pd.DataFrame:

        fp = self._train_h5_fp if not self._debug else self._sample_h5_fp
        
        with pd.HDFStore(fp, mode = "r") as h5_storage:
            df = h5_storage.select("df", where=where)

        return df


    def get_xy(self, where:str) -> Tuple[np.ndarray, np.ndarray]:
        df = self._get_train_eras( where = where)
        return df[self.features].to_numpy(), df[self.targets].to_numpy()



    def _load_data(self, split:str):
        spinner = Halo(text=f'Loading {split} data', spinner='dots')
        spinner.start(f'Loading {split} data')
        df = pd.read_parquet(getattr(self, f"_{split}_fp"))
        spinner.succeed()
        return df


    def _new_round_exist(self) -> bool:

        prev_round = self._info.get("round", None)
        new_round_exists = prev_round != self._current_round

        if new_round_exists:
            self._info["round"] = self._current_round
            self._update_info()
        
        return new_round_exists


    def _data_exist(self) -> bool:
        # check if the files exist
        return (
            os.path.exists(self._train_fp)
            and os.path.exists(self._val_fp)
            and  os.path.exists(self._tournament_fp)
        )


    def _download_data(self, force_download :bool) -> None:

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

    
    def _create_train_hdf(self, df: pd.DataFrame) -> None:
        spinner = Halo(text=f'Turning training data into  h5py', spinner='dots')
        spinner.start()

        
        #save number of eras
        self._info["n_train_eras"] = df["era"].nunique()

        # save shape
        self._info["shape"] = df.shape

        # set index to eras
        df["era"] = df["era"].astype(int)
        df = df.set_index("era")


        # create hdf file
        df.to_hdf(self._train_h5_fp, key='df', mode='w', format='table')

        #end spinner
        spinner.succeed()


    def _create_sample_data(self) -> None:

        spinner = Halo(text="creating sample data")
        spinner.start()

        rows = np.random.random_integers(low = 0, high = self._info["n_train_eras"], size = 4)


    def _get_column_values(self):
        df = self._get_train_eras(where = "index = 1")
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
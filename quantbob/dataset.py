
#basics
import os
import pandas as pd
from pathlib import Path
import pytz
from datetime import datetime

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

        # read the data into dataframes
        self.__read_data()


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


    def __read_data(self):
        spinner = Halo(text='Downloading Numerai Data', spinner='dots')

        spinner.start('Reading parquet data')
        training_data = pd.read_parquet(self._train_fp)
        validation_data = pd.read_parquet(self._val_fp)
        tournament_data = pd.read_parquet(self._tournament_fp)
        
        #example_preds = pd.read_parquet(f'/tmp/example_predictions.parquet')
        #validation_preds = pd.read_parquet('/tmp/example_validation_predictions.parquet')
        spinner.succeed()






    # def __getitem__(self, key):
    #     data_dict = {}
    #     with h5py.File(self.h5py_fp, "r") as data:
            
    #         for key in data:
    #             data_dict[key] = data[key][key]

    #     return data_dict


    # def __len__(self):
    #     return self._size 


    # # def __setup_h5py(self):
    # #     self.h5py_f = h5py.File(os.path.join(self.root_dir, "data.h5py"), 'w')


    # def __extract_data(self, split=None):
    #     training_data = pd.read_csv(os.path.join(self.path_to_data, "numerai_training_data.csv"))
    #     training_data["split"] = None
    #     test_data = pd.read_csv(os.path.join(self.path_to_data, "numerai_tournament_data.csv"))
    #     training_data["split"] = "test"

    #     train_idxs, val_idxs = train_test_split(training_data.index.to_numpy(), test_size=0.3)

    #     training_data.loc[train_idxs,"split"] = "train"
    #     training_data.loc[val_idxs,"split"] = "train"

    #     big_df = pd.concat(training_data, test_data, ignore_index=True)

    #     pass


    # def __store_data(self):
    #     features, timesteps, targets, ids, splits =  self.__extract_data(, split=0.3)
    #     h5py_fo = h5py.File(self.h5py_fp, 'w')
    #     h5py_fo.create_dataset("features",  data=features.shape,    dtype=features.dtype)
    #     h5py_fo.create_dataset("timesteps", data=timesteps,         dtype=np.int16)
    #     h5py_fo.create_dataset("targets",   data=targets,           dtype=np.int16)
    #     h5py_fo.create_dataset("ids",       data=ids,               dtype=np.int16)

    #     self._size = features.shape[0]



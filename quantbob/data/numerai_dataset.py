
#basics
from typing import Tuple, List
import os
import pandas as pd
import json
import shutil

from tqdm.auto import tqdm
import pyarrow as pa
from pyarrow import parquet
from halo import Halo

from quantbob.datamodule import DataModule

# numerapi
from numerapi import NumerAPI

# setup the napi, so we can import it to
# other modules
napi = NumerAPI()


class NumerAIDataset:

    """
    Loads the numerai dataset
    
    """

    def __init__(self, debug:bool = False, force_download :bool =  False) -> None:

        # _debug
        self._debug = debug
        
        # set current round
        self._current_round = napi.get_current_round()

        # setup directory name
        self._path_to_data = f"/tmp/numerai_{self._current_round}"
        
        # if we want to force a new download we can
        # just delete the data that exists
        if force_download and os.path.exists(self._path_to_data):
            shutil.rmtree(self._path_to_data)
        
        # make directory
        os.makedirs(self._path_to_data, exist_ok = True)

        # set cv folders
        self._path_to_cv_data = os.path.join(self._path_to_data, "csv_data")
        os.makedirs(self._path_to_cv_data, exist_ok = True)

        # set datafiles
        ## setup directory
        self._train_fp : str = os.path.join(self._path_to_data, "numerai_training_data.parquet")
        self._val_fp : str = os.path.join(self._path_to_data, "numerai_validation_data.parquet")
        self._tournament_fp : str  = os.path.join(self._path_to_data, "numerai_tournament_data.parquet")
        self._sample_fp : str  = os.path.join(self._path_to_data, "train_sample.parquet") # for debugging quickly

        # download data
        self._download_data()
            
        if self._debug:
            
            if not os.path.exists(self._sample_fp):
                self._create_sample_data()
    
            self._train_fp =  self._sample_fp
            self._path_to_cv_data = os.path.join(self._path_to_data, "csv_data_sample")
            os.makedirs(self._path_to_cv_data, exist_ok = True)

    @property
    def current_round(self):
        return self._current_round

    @Halo(text='Loading Train Data', spinner='dots')
    def get_train_data(self):
        return pd.read_parquet(self._train_fp)

    @Halo(text='Loading Val Data', spinner='dots')
    def get_val_data(self):
        return pd.read_parquet(self._val_fp)

    @Halo(text='Loading Tournament Data', spinner='dots')
    def get_tournament_data(self):
        return pd.read_parquet(self._tournament_fp)


    def _data_exist(self) -> bool:
        # check if the files exist
        return (
            os.path.exists(self._train_fp)
            and os.path.exists(self._val_fp)
            and  os.path.exists(self._tournament_fp)
        )


    def _download_data(self) -> None:

        """
        downloading the current data using the numerapi

        from https://github.com/numerai/example-scripts/blob/master/example_model.py
        
        """
        # read in all of the new datas
        # tournament data and example predictions change every week so we specify the round in their names
        # training and validation data only change periodically, so no need to download them over again every single week
        
        if not os.path.exists(self._train_fp):
            napi.download_dataset("numerai_training_data.parquet", self._train_fp)
            
        if not os.path.exists(self._val_fp):
            napi.download_dataset("numerai_validation_data.parquet", self._val_fp)
        
        if not os.path.exists(self._tournament_fp):
            napi.download_dataset("numerai_tournament_data.parquet", self._tournament_fp)
            

        #napi.download_dataset("example_predictions.parquet", os.path.join(path_to_data, "example_predictions.parquet"))
        #napi.download_dataset("example_validation_predictions.parquet", os.path.join(path_to_data, "example_validation_predictions.parquet"))


    def _create_sample_data(self) -> None:
        """
        creates a new parquet file with from the 50k first row in the training dataset.
        
        we use pyarrow to load the file content by batch be able to run quantbob in debug mode
        on machines with lower memory than 32GB
        """
        pa_file = parquet.ParquetFile(self._train_fp)
        for batch in pa_file.iter_batches(batch_size=120000):
            df = pa.Table.from_batches([batch]).to_pandas()         
            break
         
        df.to_parquet(self._sample_fp)
        
    
    def get_cv_datamodules(
        self, 
        n_folds : int, 
        remove_leakage : bool = True, 
        train_val_time_gap : bool = False
        ) -> List[DataModule]:
        
        # set eras as index
        df = self.get_train_data()
        
        if self._debug:
            n_folds = 3
        
        # filter leakage eras
        df["era"] = df["era"].astype(int)
        eras = [i for i in range(1, df["era"].nunique(), 5 if remove_leakage else 1)]
        df.set_index(["era"], inplace=True)
        df = df.loc[eras, :]

        # number of eras
        n_eras = len(eras)
                
        # setting split sizes
        split_size = n_eras // n_folds
        train_split_size = max((split_size // 3), 1) * 2 # 33% as test
        
        # split files
        datamodules = []
                        
        # creating parquet files
        split_id = 0
        for i in tqdm(range(0, n_eras, split_size), 
                    total = n_folds, 
                    desc = "Creating parquet split files"):
            
            # splitting eras            
            split_eras = eras[i:i+split_size]
            train_eras = split_eras[:train_split_size]
            test_eras = split_eras[train_split_size:]
            
            #splitting training data
            train_file_fp = os.path.join(self._path_to_cv_data, f"train_{split_id}.parquet")
            #print(df.loc[train_eras, :])
            df.loc[train_eras, :].to_parquet(train_file_fp)

            #splitting test data
            test_file_fp = os.path.join(self._path_to_cv_data, f"test_{split_id}.parquet")
            df.loc[test_eras, :].to_parquet(test_file_fp)

            # adding files 
            datamodules.append(DataModule(train_file_fp, test_file_fp))
            
            # update split_id
            split_id += 1
            
        return datamodules
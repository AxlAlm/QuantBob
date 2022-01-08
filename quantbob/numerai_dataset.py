
#basics
from typing import Tuple, List
import os
import pandas as pd
import json
from tqdm.auto import tqdm

# halo
from halo import Halo

# napi
from quantbob import napi


        # return ([c for c in df.columns if "feature_" in c], 
        #         [c for c in df.columns if "target_" in c])


class ParquetCVReader:
    
    def __init__(self, parquet_cv_files: List[Tuple]) -> None:
        self._parquet_cv_files = parquet_cv_files
        
    
    def __iter__(self) -> pd.DataFrame:
        
        for (train_fp, test_fp) in self._parquet_cv_files:
            
            train_df = pd.read_parquet(train_fp)
            test_df = pd.read_parquet(test_fp)
            
            yield  train_df, test_df


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


        # only contain info of current round atm
        self._info_fp : str = os.path.join(self._path_to_data, "info.json")
        self._info = self._load_info()

        # set datafiles
        self._train_fp : str = os.path.join(self._path_to_data, "numerai_training_data.parquet")
        self._val_fp : str = os.path.join(self._path_to_data, "numerai_validation_data.parquet")
        self._tournament_fp : str  = os.path.join(self._path_to_data, "numerai_tournament_data.parquet")

        # set current round
        self._current_round = napi.get_current_round()

        # check if there is new data
        if not self._data_exist() or self._new_round_exist() or force_download:

            # download data
            self._download_data(force_download = force_download)

        ## update the info
        self._update_info()

    @property
    def current_round(self):
        return self._current_round


    def get_train_data(self):
        return self._load_data("train")


    def get_val_data(self):
        return self._load_data("val")


    def get_tournament_data(self):
        return self._load_data("tournament")


    def _load_info(self) -> dict:

        if not os.path.exists(self._info_fp):
            return {}

        with open(self._info_fp, "r") as f:
            info = json.load(f)

        return info


    def _update_info(self) -> None:
        with open(self._info_fp, "w") as f:
            json.dump(self._info, f)


    def _load_data(self, split:str):
        spinner = Halo(text=f'Loading {split} data', spinner='dots')
        spinner.start()
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


    def create_cvs(self, n_folds : int, method = "time_splits", remove_leakage : bool = True) -> ParquetCVReader:
     
        # set eras as index
        df = self.get_train_data()
        
        ## setup directory
        self._path_to_cv_data = "/tmp/numerai_data/cv_data"
        os.makedirs(self._path_to_cv_data, exist_ok = True)
                   
        if method == "time_splits":
            return self._create_time_splits( 
                                            df = df,
                                            n_folds = n_folds, 
                                            remove_leakage = remove_leakage
                                            )
        else:
            raise KeyError("Invalid cv split method")
    
    
    @Halo(text='Creating CV files', spinner='dots')
    def _create_time_splits(self, df:pd.DataFrame, n_folds : int, remove_leakage : bool) -> ParquetCVReader:
        
        # filter leakage eras
        df["era"] = df["era"].astype(int)
        n_eras = df["era"].nunique()
        eras = [i for i in range(1, n_eras, 5 if remove_leakage else 1)]
        df = df.loc[df["era"].isin(eras)]

        # setting split sizes
        split_size = n_eras // n_folds
        train_split_size = (split_size // 3) * 2 # 33% as test

        # split files
        split_files : List[Tuple] = []
                        
        # creating parquet files
        split_id = 0
        for i in tqdm(range(0, n_eras, split_size), 
                    total = n_folds, 
                    desc = "Creating parquet split files"):
            
            # splitting eras            
            split_eras = eras[i:i+i]
            train_eras = split_eras[:train_split_size]
            test_eras = split_eras[train_split_size:]

            #splitting training data
            train_file_fp = os.path.join(self._path_to_cv_data, f"train_{split_id}.parquet")
            df.loc[df.isin(train_eras)].to_parquet(train_file_fp)

            #splitting test data
            test_file_fp = os.path.join(self._path_to_cv_data, f"test_{split_id}.parquet")
            df.loc[df.isin(test_eras)].to_parquet(test_file_fp)

            # adding files 
            split_files.append((train_file_fp, test_file_fp))
            

        return ParquetCVReader(parquet_cv_files = split_files)
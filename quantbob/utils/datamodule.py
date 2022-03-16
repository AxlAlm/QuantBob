
from typing import Tuple

import pandas as pd
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from quantbob.utils.dataset import Dataset



class DataModule:
    
    def __init__(self, train_parquet_fp:str, val_parquet_fp:str) -> None:
        self._train_parquet_fp = train_parquet_fp
        self._val_parquet_fp = val_parquet_fp

    def to_pldm(self, batch_size:int) -> "PLDataModule":
        return PLDataModule(batch_size)
    
    def to_dask(self) -> Tuple:
        return PLDataModule()
    
    def _get_xy(self, df:pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = df.filter(regex='^feature_', axis=1).to_numpy()
        y = df.filter(regex='^target_', axis=1).to_numpy()
        return X, y
            
    def train_xy(self) -> np.ndarray:
        df = pd.read_parquet(self._train_parquet_fp)
        return self._get_xy(df)
            
    def val(self) -> np.ndarray:
        df = pd.read_parquet(self._val_parquet_fp).to_numpy()
        return self._get_xy(df)   
    

class DaskDataModule:
        
    def _get_xy(self, df:pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = df.filter(regex='^feature_', axis=1).to_numpy()
        y = df.filter(regex='^target_', axis=1).to_numpy()
        return X, y
            
    def train_xy(self) -> np.ndarray:
        df = pd.read_parquet(self._train_parquet_fp)
        return self._get_xy(df)
            
    def val(self) -> np.ndarray:
        df = pd.read_parquet(self._val_parquet_fp).to_numpy()
        return self._get_xy(df)   
    

class PLDataModule(pl.LightningDataModule):
    
    def __init__(self, train_parquet_fp:str, val_parquet_fp:str, batch_size:int) -> None:
        super().__init__()
        self._train_parquet_fp = train_parquet_fp
        self._val_parquet_fp = val_parquet_fp
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            Dataset(self.train()), 
            batch_size=self.batch_size, 
            shuffle=True, 
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            Dataset(self.val()), 
            batch_size=self.batch_size, 
            shuffle=True, 
            pin_memory=True
        )
        
        



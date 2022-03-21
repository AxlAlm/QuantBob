
from typing import Tuple

import dask.dataframe as dd
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch

class Dataset(torch.utils.data.Dataset):
    
  def __init__(self, features, targets):
        self._features = torch.Tensor(features)
        self._targets = torch.Tensor(targets)

  def __len__(self):
        return len(self._features)

  def __getitem__(self, index):
        return self._features[index], self._targets[index]


class DataModule(pl.LightningDataModule):
    
    def __init__(self, train_parquet_fp:str, val_parquet_fp:str) -> None:
        self._train_parquet_fp = train_parquet_fp
        self._val_parquet_fp = val_parquet_fp
    
    def _get_xy(self, df:pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = df.filter(regex='^feature_', axis=1).to_numpy()
        y = df.filter(regex='^target_', axis=1).to_numpy()
        return X, y
            
    def get_xy(self, split:str, dask:bool=False) -> np.ndarray:
        
        if split == "train":
            fp = self._train_parquet_fp
        elif split == "val":
            fp = self._val_parquet_fp
        else:
            raise ValueError(f"'{split}' is not a supported split. Try 'train' or 'val'")
        
        if dask:
            df = pd.read_parquet(fp)
        else:
            df = dd.read_parquet(fp)


        X = df.filter(regex='^feature_', axis=1).to_numpy()
        y = df.filter(regex='^target_', axis=1).to_numpy()
        
        return self._get_xy(df)
             
    
    def train_dataloader(self, batch_size:int) -> DataLoader:
        return DataLoader(
            Dataset(self.get_xy("train")), 
            batch_size=batch_size, 
            shuffle=True, 
            pin_memory=True
        )

    def val_dataloader(self, batch_size:int)  -> DataLoader:
        return DataLoader(
            Dataset(self.get_xy("val")), 
            batch_size=batch_size, 
            shuffle=True, 
            pin_memory=True
        )
        
        
        
        

# class DaskDataModule:
        
#     def _get_xy(self, df:pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
#         X = df.filter(regex='^feature_', axis=1).to_numpy()
#         y = df.filter(regex='^target_', axis=1).to_numpy()
#         return X, y
            
#     def train_xy(self) -> np.ndarray:
#         df = pd.read_parquet(self._train_parquet_fp)
#         return self._get_xy(df)
            
#     def val(self) -> np.ndarray:
#         df = pd.read_parquet(self._val_parquet_fp).to_numpy()
#         return self._get_xy(df)   
    

# class PLDataModule(pl.LightningDataModule):
    
#     def __init__(self, train_parquet_fp:str, val_parquet_fp:str, batch_size:int) -> None:
#         super().__init__()
#         self._train_parquet_fp = train_parquet_fp
#         self._val_parquet_fp = val_parquet_fp
#         self.batch_size = batch_size

#     def train_dataloader(self) -> DataLoader:
#         return DataLoader(
#             Dataset(self.train()), 
#             batch_size=self.batch_size, 
#             shuffle=True, 
#             pin_memory=True
#         )

#     def val_dataloader(self) -> DataLoader:
#         return DataLoader(
#             Dataset(self.val()), 
#             batch_size=self.batch_size, 
#             shuffle=True, 
#             pin_memory=True
#         )
        
        



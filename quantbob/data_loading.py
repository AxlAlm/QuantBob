
#basics
import os
import pandas as pd
from pathlib import Path

#h5py
import h5py

#sklearn
from sklearn.model_selection import train_test_split






class NumeraiDataloader(ptl.LightningDataModule):

    def __init__(self, 
                path_to_data, 
                batch_size:int=32, 
                root_dir:str=os.path.join(Path.home(),".quantbob")
                ):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.path_to_data = path_to_data
        self.h5py_fp = os.path.join(self.root_dir, "data.h5py")

        if os.path.exists(self.h5py_fp):
            with h5py.File(self.h5py_fp, "r") as data:
                self._size = data["ids"].shape[0]
        else:
            self.__store_data()


    def __getitem__(self, key):
        data_dict = {}
        with h5py.File(self.h5py_fp, "r") as data:
            
            for key in data:
                data_dict[key] = data[key][key]

        return data_dict


    def __len__(self):
        return self._size 


    def __setup_h5py(self):
        self.h5py_f = h5py.File(os.path.join(self.root_dir, "data.h5py"), 'w')


    def __extract_data(self, split=None):
        training_data = pd.read_csv(os.path.join(self.path_to_data, "numerai_training_data.csv"))
        training_data["split"] = None
        test_data = pd.read_csv(os.path.join(self.path_to_data, "numerai_tournament_data.csv"))
        training_data["split"] = "test"

        train_idxs, val_idxs = train_test_split(training_data.index.to_numpy(), test_size=0.3)

        training_data.loc[train_idxs,"split"] = "train"
        training_data.loc[val_idxs,"split"] = "train"

        big_df = pd.concat(training_data, test_data, ignore_index=True)

        pass


    def __store_data(self):
        features, timesteps, targets, ids, splits =  self.__extract_data(, split=0.3)

        
        h5py_fo = h5py.File(self.h5py_fp, 'w')
        h5py_fo.create_dataset("features",  data=features.shape,    dtype=features.dtype)
        h5py_fo.create_dataset("timesteps", data=timesteps,         dtype=np.int16)
        h5py_fo.create_dataset("targets",   data=targets,           dtype=np.int16)
        h5py_fo.create_dataset("ids",       data=ids,               dtype=np.int16)

        self._size = features.shape[0]


    def train_dataloader(self):
        # ids are given as a nested list (e.g [[42, 43]]) hence using lambda x:x[0] to select the inner list.
        sampler = BatchSampler(self.splits[self.split_id]["train"], batch_size=self.batch_size, drop_last=False)
        return DataLoader(self, num_workers=multiprocessing.cpu_count())


    def val_dataloader(self):
        # ids are given as a nested list (e.g [[42, 43]]) hence using lambda x:x[0] to select the inner list.
        sampler = BatchSampler(self.splits[self.split_id]["val"], batch_size=self.batch_size, drop_last=False)
        return DataLoader(self, sampler=sampler, collate_fn=lambda x:x[0], num_workers=multiprocessing.cpu_count())


    def test_dataloader(self):
        # ids are given as a nested list (e.g [[42, 43]]) hence using lambda x:x[0] to select the inner list.
        sampler = BatchSampler(self.splits[self.split_id]["test"], batch_size=self.batch_size, drop_last=False)
        return DataLoader(self, sampler=sampler, collate_fn=lambda x:x[0], num_workers=multiprocessing.cpu_count())



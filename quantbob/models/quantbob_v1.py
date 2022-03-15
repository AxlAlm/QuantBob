
import argparse
import os
from typing import List
from typing import Optional

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms


PERCENT_VALID_EXAMPLES = 0.1
BATCHSIZE = 128
CLASSES = 10
EPOCHS = 10
DIR = os.getcwd()


class V1(pl.LightningModule):
    def __init__(self, dropout: float, output_dims: List[int]):
        super().__init__()
        pass
        
    @classmethod
    def from_optuna(cls, trial):

        hyperparameters = {
            
        }
        return cls(**hyperparameters)


    def forward(self, data: torch.Tensor) -> torch.Tensor:
        pass

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        pass
    
    
    def validation_step(self, batch, batch_idx: int) -> None:
        pass
        self.log("val_acc", accuracy, sync_dist=True)
        self.log("hp_metric", accuracy, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.model.parameters())


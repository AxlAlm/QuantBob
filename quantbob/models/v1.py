
import argparse
import os
from typing import List

import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim

class V1(pl.LightningModule):
    def __init__(self, dropout: float, output_dims: List[int]):
        super().__init__()
        pass
    

    def get_optuna_search_space(self) -> dict:
        pass
        
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


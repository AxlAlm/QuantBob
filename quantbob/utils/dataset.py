


import torch

class Dataset(torch.utils.data.Dataset):
    
  def __init__(self, features, targets):
        self._features = torch.Tensor(features)
        self._targets = torch.Tensor(targets)

  def __len__(self):
        return len(self._features)

  def __getitem__(self, index):
        return self._features[index], self._targets[index]
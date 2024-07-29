import torch
from datasets import Dataset

from typing import Tuple

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: Dataset, *args, **kwargs):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        
        x = self.data[idx]['signal']
        y = self.data[idx]['label']

        return x, y
import torch
from datasets import Dataset

from typing import Tuple

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: Dataset, *args, **kwargs):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.data[idx]
        first = list(data.keys())[0]
        second = list(data.keys())[1]

        return data[first].type(torch.float32), data[second].type(torch.float32)
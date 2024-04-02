import torch
import datasets
from datasets import Dataset

from typing import Tuple

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: Dataset, *args, **kwargs):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.data[idx]

        label = sample['label']
        if label == 1:
            label = 0
        elif label == 2:
            label = 1
        elif label == 3:
            label = 0
        elif label == 4:
            label = 0

        x = torch.tensor(sample['features'], dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.float32)

        return x, y
import torch
from datasets import Dataset, load_dataset

from typing import Tuple

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: Dataset, window: int, *args, **kwargs):
        self.data = data.with_format("torch")
        self.window = window

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx + self.window > len(self.data):
            raise StopIteration
        
        window = self.data[idx: idx + self.window]

        x = window['signal']
        y = torch.tensor(1, dtype=torch.float32) if torch.mode(window['label'], 0)[0] == 1 else torch.tensor(0, dtype=torch.float32)

        return x, y
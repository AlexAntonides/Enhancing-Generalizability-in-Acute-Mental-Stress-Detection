import torch
from datasets import Dataset, load_dataset

from typing import Tuple

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: Dataset, window: int, *args, **kwargs):
        self.data = data.with_format("torch")
        self.window = window

    def __len__(self) -> int:
        return len(self.data) // self.window

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # Fixed Window
        idx = idx * self.window
         
        if idx + self.window > len(self.data):
            raise StopIteration
        
        window = self.data[idx: idx + self.window]

        x = window['signal']
        y = torch.tensor(1, dtype=torch.long) if torch.mode(window['label'], 0)[0] == 1 else torch.tensor(0, dtype=torch.long)

        return torch.unsqueeze(x, 0), y
        # return torch.unsqueeze(x, -1), y
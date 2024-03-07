import torch
from datasets import Dataset
import torchvision.transforms as transforms 

from typing import Tuple

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: Dataset, *args, **kwargs):
        self.data = data

        self.transform = transforms.Compose([ 
            transforms.PILToTensor() 
        ]) 

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.data[idx]
        return self.transform(data['pixel_values']).type(torch.float32), torch.tensor(data['label'], dtype=torch.float32)
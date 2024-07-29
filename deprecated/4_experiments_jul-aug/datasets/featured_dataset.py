import torch
from datasets import Dataset, load_dataset

import numpy as np
from typing import Tuple

X_labels =  [
    "hrv_rms",
    "hr_mean",
    "hr_min",
    "hr_max",
    "rr_mean",
    "rr_min",
    "rr_max",
    "rr_std",
    "nn50",
    "pnn50",
    "rmssd",
    "lf",
    "hf",
    "vhf",
    "uhf",
    "tp",
    "lp_hf",
    "lp_vhf",
    "lp_uhf",
    "w",
    "wmax",
    "wen",
    "CVNN",
    "CVSD",
    "MedianNN",
    "MadNN",
    "MCVNN",
    "IQRNN",
    "SDRMSSD",
    "Prc20NN",
    "Prc80NN",
    "pNN20",
    "HTI",
    "TINN",
    "twa",
]
y_label = 'target'

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: Dataset, *args, **kwargs):
        self.data = data.with_format("torch")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data.select_columns(X_labels)[idx]
        y = self.data[idx][y_label]

        X = torch.stack(list(X.values()))

        # X = X.to(dtype=torch.float32)
        # y = y.to(dtype=torch.float32)
        return X, y
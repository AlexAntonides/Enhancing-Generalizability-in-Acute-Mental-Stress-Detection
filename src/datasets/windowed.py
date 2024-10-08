import torch
from datasets import Dataset, load_dataset

class WindowedDataset(torch.utils.data.IterableDataset):
    def __init__(self, data: Dataset, window: int, train_mean=None, train_std=None, *args, **kwargs):
        self.data = data.with_format("torch")
        
        self.window = window

        self.train_mean = train_mean
        self.train_std = train_std
        
    def __len__(self):
        return len(self.data) // self.window

    def __iter__(self):
        # For each window in the dataset,
        for start_idx in range(0, len(self.data), self.window):
            # If the window is too short, skip it
            if start_idx + self.window > len(self.data):
                break
            window = self.data[start_idx: start_idx + self.window]

            # If the window contains multiple labels, skip it
            if len(window['label'].unique()) != 1:
                continue
            else:
                x = window['signal']
                y = window['label'].unique()[0]
                
                # Standardize the data when necessary (only in training and validation)
                if self.train_mean is not None and self.train_std is not None:
                    x = (x - self.train_mean) / self.train_std

                yield x, y
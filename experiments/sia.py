import torch
torch.cuda.is_available()

import numpy as np
import pandas as pd
from torch.utils.data import Dataset as TorchDataset
from tabulate import tabulate

import lightning as L

from pyarrow.parquet import ParquetDataset

from pyarrow.parquet import ParquetDataset
from sia.utils import get_file_paths

pd.options.mode.chained_assignment = None
target_labels = ['TA', 'SSST_Sing_countdown', 'Pasat', 'Raven', 'TA_repeat', 'Pasat_repeat']

class Dataset(TorchDataset):#IterableDataset):
    def __init__(self, name: str, window: int = 1000):
        self.name = name
        self.window = window
        self.dataset = None

    def attach(self, arg: str):
        if isinstance(arg, str):
            files = get_file_paths(arg)
            self.dataset = ParquetDataset(files)
        elif hasattr(arg, "__len__"):
            files = arg
            self.dataset = ParquetDataset(files)
        else: 
            raise TypeError("The argument type is not supported")
        
        return self
    
    def __len__(self):
        return sum(p.count_rows() for p in self.dataset.fragments)

    def __getitem__(self, idx):
        i = idx
        for piece in self.dataset.fragments:
            if i - piece.count_rows() < 0:
                window = piece.take(list(range(idx, idx+self.window)), columns=['ECG_Clean', 'category'])
                break
            else:
                i -= piece.count_rows()
        
        if window is None:
            raise IndexError("Index out of range")
        
        signal = window['ECG_Clean'].to_numpy()
        label = window['category'].to_numpy()
        label[np.isin(label, target_labels)] = 1
        label[~np.isin(label, target_labels)] = 0
        
        signal = torch.tensor(signal)
        label = torch.tensor(label.astype(int))

        return signal, torch.tensor(1) if torch.mode(label, 0)[0] == 1 else torch.tensor(0)
        
    def __repr__(self):
        data = []

        data.append(["name", self.name])

        if len(self.dataset.fragments) > 0:
            data.append(["files", len(self.dataset.fragments)])
            
            data.append(["length", [f.count_rows() for f in self.dataset.fragments]])

        return tabulate(data, tablefmt="fancy_grid")


import glob
from sklearn.model_selection import train_test_split
participants = glob.glob("./data/parquet/*.parquet")
train_participants, test_participants = train_test_split(participants,test_size=0.99) #test_size=0.2)

ds_train = Dataset("Stress-in-Action")
ds_train.attach(train_participants)
ds_train

ds_test = Dataset("Stress-in-Action")
ds_test.attach(test_participants)
ds_test

from torch.utils.data import DataLoader

train_dataloader = DataLoader(ds_train, batch_size=32, shuffle=False, drop_last=True, num_workers=19, pin_memory=True)
test_dataloader = DataLoader(ds_test, batch_size=32, shuffle=False, drop_last=True, pin_memory=True)

import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="stress-in-action",
    
    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "SiA",
        "epochs": 11,
    }
)

from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn

class Test(L.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(1000, 10).double(),
            nn.Tanh(),
            nn.Linear(10, 1).double(),
            nn.Softmax(1),
        )

        self.layers.cuda(0)

    def forward(self, x):
        embedding = self.layers(x)
        return embedding

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=0.02)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        wandb.log({"loss": loss})
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        wandb.log({"val_loss": loss})
        return loss

model = Test()
model.cuda()
trainer = L.Trainer(max_epochs=11, accelerator="gpu", devices="auto", strategy="auto", profiler="simple")
trainer.fit(model, train_dataloader, test_dataloader)

wandb.finish()
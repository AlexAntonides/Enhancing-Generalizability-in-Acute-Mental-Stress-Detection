import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np 
import pandas as pd

from datasets import Dataset as HFDataset, load_dataset

import torch
torch.set_float32_matmul_precision('medium')

from glob import glob 

import torch

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: HFDataset, window: int = 1000):
        self.data = data
        self.window = window

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        window = self.data[idx: idx + self.window]

        x = window['signal']
        y = torch.tensor(1, device='cuda') if torch.mode(window['label'], 0)[0] == 1 else torch.tensor(0, device='cuda')

        return x, y
    
from sklearn.model_selection import train_test_split

participants = glob("./data/test_2/*.csv")
train_participants, test_participants = train_test_split(participants, test_size=0.2)
train_participants, val_participants = train_test_split(train_participants, test_size=0.25)

from torch.utils.data import DataLoader
import lightning as L
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

from torchmetrics.functional import accuracy
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, OptimizerLRScheduler
from torch import nn

class Test(L.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(1000, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Softmax(-1),
        )
        
    def prepare_data(self):
        data = load_dataset('csv', data_files={
            'train': train_participants[:1],
            'val': val_participants[:1],
            'test': test_participants[:1]
        }, column_names=['signal', 'label'])
        data = data.with_format('torch', device=self.device)

        self.train_dataset = Dataset(data['train'])
        self.val_dataset = Dataset(data['val'])
        self.test_dataset = Dataset(data['test'])

    def forward(self, x):
        embedding = self.layers(x)
        return embedding
    
    def training_step(self, batch, batch_idx):
        loss, acc = self._step(batch, batch_idx)
        wandb.log({"accuracy": acc, "loss": loss})
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(batch, batch_idx)
        wandb.log({"val_accuracy": acc, "val_loss": loss})
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, acc = self._step(batch, batch_idx)
        wandb.log({"test_accuracy": acc, "test_loss": loss})
        return loss

    def _step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = nn.functional.cross_entropy(y_hat, y)
        acc = accuracy(y_hat.squeeze(), y, task='binary')
        return loss, acc
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.parameters(), lr=0.02)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=128)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=128)
    
    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=128)

model = Test()
model.cuda()

trainer = L.Trainer(
    max_epochs=11, 
    accelerator="cpu", 
    devices="auto", 
    strategy="auto", 
    profiler="simple"
)

from lightning.pytorch.tuner import Tuner
tuner = Tuner(trainer)

trainer.fit(
    model=model
)

wandb.finish()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np 
import pandas as pd

from datasets import Dataset as HFDataset, load_dataset

import torch
torch.set_float32_matmul_precision('medium')

from glob import glob 

import torch
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: HFDataset, window: int = 1000):
        self.data = data
        self.window = window

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx + self.window > len(self.data):
            raise StopIteration
        
        window = self.data[idx: idx + self.window]

        x = window['signal']
        y = torch.tensor(1, dtype=torch.float32) if torch.mode(window['label'], 0)[0] == 1 else torch.tensor(0, dtype=torch.float32)

        return x, y
    
from sklearn.model_selection import train_test_split

participants = glob("./data/test_2/*.csv")
train_participants, test_participants = train_test_split(participants, test_size=0.2)
train_participants, val_participants = train_test_split(train_participants, test_size=0.25)

from torch.utils.data import DataLoader
import lightning as L

import wandb

import torchmetrics
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn

class Test(L.LightningModule):
    def __init__(self, batch_size, learning_rate, num_workers):
        super().__init__()

        self.save_hyperparameters()
        
        self.layers = nn.Sequential(
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Softmax(-1),
        )
        
    def prepare_data(self):
        self.data = load_dataset('csv', data_files={
            'fit': train_participants,
            'validate': val_participants,
            'test': test_participants
        }, column_names=['signal', 'label'], num_proc=8)
        self.data = self.data.with_format('torch', device=self.device)

    def setup(self, stage):
        self.dataset = Dataset(self.data[stage])
        self.accuracy = torchmetrics.classification.BinaryAccuracy()
        self.f1score = torchmetrics.classification.BinaryF1Score()

    def forward(self, x):
        embedding = self.layers(x)
        return embedding.squeeze()
    
    def training_step(self, batch, batch_idx):
        y, y_hat = self._step(batch, batch_idx)
        
        step_loss = nn.functional.binary_cross_entropy(y_hat, y)
        self.accuracy.update(y_hat.squeeze(), y)
        self.f1score.update(y_hat.squeeze(), y)

        wandb.log({"accuracy": self.accuracy.compute(), "loss": step_loss, "f1": self.f1score.compute()})
        return step_loss

    def validation_step(self, batch, batch_idx):
        y, y_hat = self._step(batch, batch_idx)

        step_loss = nn.functional.binary_cross_entropy(y_hat, y)
        self.accuracy.update(y_hat.squeeze(), y)
        self.f1score.update(y_hat.squeeze(), y)

        wandb.log({"val_accuracy": self.accuracy.compute(), "val_loss": step_loss, "val_f1": self.f1score.compute()})
        return step_loss
    
    def test_step(self, batch, batch_idx):
        y, y_hat = self._step(batch, batch_idx)

        step_loss = nn.functional.binary_cross_entropy(y_hat, y)
        self.accuracy.update(y_hat.squeeze(), y)
        self.f1score.update(y_hat.squeeze(), y)

        wandb.log({"test_accuracy": self.accuracy.compute(), "test_loss": step_loss, "test_f1": self.f1score.compute()})
        return step_loss

    def _step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return y, y_hat
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True)

batch_size = 1024
learning_rate = 0.8317637711026709
num_workers = 16
epochs = 11

wandb.init(
    # set the wandb project where this run will be logged
    project="stress-in-action",
    
    # track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "architecture": "Test",
        "dataset": "SiA",
        "epochs": epochs,
    }
)

model = Test(batch_size=batch_size, learning_rate=learning_rate, num_workers=num_workers)

trainer = L.Trainer(
    max_epochs=epochs, 
    accelerator="gpu", 
    devices="auto", 
    strategy="auto", 
    profiler="simple",
    default_root_dir="./checkpoints/test_model",
)

tuner = Tuner(trainer)
# tuner.scale_batch_size(model)
# tuner.lr_find(model)

trainer.fit(
    model=model
)

wandb.finish()
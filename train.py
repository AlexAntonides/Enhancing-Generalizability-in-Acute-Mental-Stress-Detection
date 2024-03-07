import warnings
warnings.filterwarnings("ignore")

import importlib
from argparse import ArgumentParser
from glob import glob 

import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning import LightningModule

from datasets import Dataset, load_dataset

import torchmetrics
from sklearn.model_selection import train_test_split

import wandb

def prepare_model(
    model: LightningModule, 
    data: str,
    window: int,
    batch_size: int, 
    learning_rate: float, 
    num_workers: int,
    train_participants: list[str],
    val_participants: list[str],
    test_participants: list[str],
    dataset: torch.utils.data.Dataset = None
) -> LightningModule:
    class Model(model):
        def __init__(self, batch_size: int, learning_rate: float, num_workers: int):
            self.save_hyperparameters()
            super().__init__()

        def prepare_data(self):
            try:
                self.data = load_dataset(
                    data,
                    trust_remote_code=True,
                    train_participants=train_participants, 
                    val_participants=val_participants, 
                    test_participants=test_participants,
                    num_proc=8 if len(train_participants) > 8 else len(train_participants)
                )
            except: 
                # Old code, gotta remove later.
                self.data = load_dataset('csv', data_files={
                    'fit': train_participants,
                    'validate': val_participants,
                    'test': test_participants
                }, column_names=['signal', 'label'], num_proc=8)

        def setup(self, stage):
            if dataset is not None:
                self.dataset = dataset(self.data[stage], window)
            else:
                self.dataset = self.data[stage]
            self.accuracy = torchmetrics.classification.BinaryAccuracy()
            self.f1score = torchmetrics.classification.BinaryF1Score()

        def forward(self, x):
            embedding = self.layers(x)
            return embedding.squeeze()
        
        def training_step(self, batch, batch_idx):
            y, y_hat = self._step(batch, batch_idx)
            
            step_loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
            self.accuracy.update(y_hat.squeeze(), y)
            self.f1score.update(y_hat.squeeze(), y)

            if wandb.run is not None:
                wandb.log({"accuracy": self.accuracy.compute(), "loss": step_loss, "f1": self.f1score.compute()})
            
            return step_loss

        def validation_step(self, batch, batch_idx):
            y, y_hat = self._step(batch, batch_idx)

            step_loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
            self.accuracy.update(y_hat.squeeze(), y)
            self.f1score.update(y_hat.squeeze(), y)

            if wandb.run is not None:
                wandb.log({"val_accuracy": self.accuracy.compute(), "val_loss": step_loss, "val_f1": self.f1score.compute()})
            
            return step_loss
        
        def test_step(self, batch, batch_idx):
            y, y_hat = self._step(batch, batch_idx)

            step_loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
            self.accuracy.update(y_hat.squeeze(), y)
            self.f1score.update(y_hat.squeeze(), y)

            if wandb.run is not None:
                wandb.log({"test_accuracy": self.accuracy.compute(), "test_loss": step_loss, "test_f1": self.f1score.compute()})
            
            return step_loss

        def _step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            return y, y_hat
        
        def configure_optimizers(self) -> L.pytorch.utilities.types.OptimizerLRScheduler:
            return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        
        def train_dataloader(self):
            return DataLoader(self.dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True)

        def val_dataloader(self):
            return DataLoader(self.dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True)
        
        def test_dataloader(self):
            return DataLoader(self.dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True)

    return Model(
        batch_size=batch_size, 
        learning_rate=learning_rate, 
        num_workers=num_workers
    )

def train(name: str, model: LightningModule, epochs: int):
    trainer = L.Trainer(
        max_epochs=epochs, 
        accelerator="auto", 
        devices="auto", 
        strategy="auto", 
        profiler="simple",
        default_root_dir=f"./checkpoints/{model_name}"
    )

    tuner = L.pytorch.tuner.Tuner(
        trainer
    )

    trainer.fit(
        model=model
    )

if __name__ == "__main__":
    parser = ArgumentParser(
        prog='Train',
        description='Trains a model with the given dataset.',
        epilog='Written by Alex Antonides.'
    )

    parser.add_argument('model', type=str, help='the model')
    parser.add_argument('data', type=str, help='the data directory', nargs='+')

    parser.add_argument('--dataset', '-d', type=str, help='the dataset', default='sia.datasets.dataset')

    parser.add_argument('--test_size', '-ts', type=float, help='the test split size', default=0.2)
    parser.add_argument('--val_size', '-vs', type=float, help='the validation split size', default=0.25)

    parser.add_argument('--project', '-p', type=str, help='the project name', default='stress-in-action')
    parser.add_argument('--ignore_wandb', '-iw', help='Ignore wandb', action='store_true')
    
    parser.add_argument('--epochs', '-e', type=int, help='the amount of epochs to train', default=11) 
    parser.add_argument('--batch_size', '-b', type=int, help='the batch size to use', default=1024) # default from tuner.scale_batch_size(model)
    parser.add_argument('--learning_rate', '-l', type=float, help='the learning rate to use', default=0.01) 
    parser.add_argument('--window', '-w', type=int, help='The amount of samples per window, where 1000hz = 1 second', default=1000)
    parser.add_argument('--n_workers', '-nw', type=int, help='The amount of workers', default=8)

    parser.add_argument('--test', '-t', help='Train with one file to speed up training for testing purposes', action='store_true')

    args = parser.parse_args()

    if isinstance(args.data, list):
        participants = args.data
    else:
        participants = glob(args.data)

    if len(participants) == 0:
        raise FileNotFoundError(f"No participants found on {args.data}:", participants)
    elif len(participants) <= 3:
        raise FileNotFoundError(f"Too few participants found on {args.data}:", participants)

    train_participants, test_participants = train_test_split(participants, test_size=args.test_size)
    train_participants, val_participants = train_test_split(train_participants, test_size=args.val_size)

    model_name = args.model.split('.')[-1]
    model_module = importlib.import_module(args.model)

    if isinstance(args.data, str):
        dataset_name = args.data.split('/')[-2]
        dataset_path = '/'.join(args.data.split('/')[:-1])
    elif isinstance(args.data, list):
        dataset_name = args.data[0].split('/')[-2]
        dataset_path = '/'.join(args.data[0].split('/')[:-1])
    else: 
        dataset_name = 'unknown'
    
    if args.dataset:
        dataset_module = importlib.import_module(args.dataset)

    model = prepare_model(
        model=model_module.Model, # assuming all models are named Model.
        data=dataset_path,
        dataset=dataset_module.Dataset if args.dataset else None,
        window=args.window,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.n_workers,
        train_participants=train_participants if not args.test else train_participants[:1],
        val_participants=val_participants if not args.test else val_participants[:1],
        test_participants=test_participants if not args.test else test_participants[:1]
    )

    if args.ignore_wandb == False:
        wandb.init(
            project=args.project,
            config={
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "num_workers": args.n_workers,
                "epochs": args.epochs,
                "architecture": model_name,
                "dataset": dataset_name
            }
        )

    train(
        model_name,
        model=model,
        epochs=args.epochs
    )
    
    if args.ignore_wandb == False:
        wandb.finish()
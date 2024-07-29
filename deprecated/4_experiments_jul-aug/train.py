import warnings
warnings.filterwarnings("ignore")

import importlib
from pathlib import Path
from argparse import ArgumentParser
from glob import glob 

import torch
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

import lightning as L
from lightning import LightningModule

from datasets import Dataset, load_dataset

import torchmetrics
from sklearn.model_selection import train_test_split
from typing import Callable

import wandb

torch.set_float32_matmul_precision('medium')

class BinaryRandomAccuracy(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_pos_labels", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        self.total += target.numel()
        self.num_pos_labels += target.sum()

    def compute(self):
        prob_positive = self.num_pos_labels / self.total
        prob_negative = 1 - prob_positive
        random_accuracy = max(prob_positive, prob_negative)
        return random_accuracy
    
def prepare_model(
    model: LightningModule, 
    data: str,
    batch_size: int, 
    learning_rate: float, 
    num_workers: int,
    ignore_torch_format: bool,
    train_participants: list[str],
    val_participants: list[str],
    test_participants: list[str],
    with_standardisation: bool = False,
    dataset: torch.utils.data.Dataset = None,
    dataset_kwargs: dict = None,
    dataset_preprocessor  = None
) -> LightningModule:
    class Model(model):
        def __init__(self, batch_size: int, learning_rate: float, num_workers: int):
            self.save_hyperparameters()
            super().__init__()

        def prepare_data(self):
            self.data = load_dataset(
                data,
                trust_remote_code=True,
                train_participants=train_participants, 
                val_participants=val_participants, 
                test_participants=test_participants,
                num_proc=self.hparams.num_workers if len(train_participants) > self.hparams.num_workers else len(train_participants)
            )

            if dataset_preprocessor is not None:
                self.data = dataset_preprocessor(self.data)

            if ignore_torch_format == False:
                self.data = self.data.with_format("torch")

            if with_standardisation:
                self.train_mean = self.data['fit']['signal'].mean()
                self.train_std = self.data['fit']['signal'].std()

        def setup(self, stage):
            if dataset is not None:
                kwargs = {}
                if with_standardisation:
                    kwargs['train_mean'] = self.train_mean
                    kwargs['train_std'] = self.train_std
                self.dataset = dataset(self.data[stage], **kwargs, **dataset_kwargs)
            else:
                self.dataset = self.data[stage]

            metrics = torchmetrics.MetricCollection([
                torchmetrics.classification.BinaryAccuracy(),
                torchmetrics.classification.BinaryF1Score(),
                torchmetrics.classification.BinaryPrecision(),
                torchmetrics.classification.BinaryAUROC(),
                BinaryRandomAccuracy()
            ])

            self.train_metrics = metrics.clone(prefix="train_")
            self.train_stats = torchmetrics.classification.BinaryStatScores()
            self.train_loss = torch.nn.BCEWithLogitsLoss()

            self.validation_metrics = metrics.clone(prefix="val_")
            self.validation_stats = torchmetrics.classification.BinaryStatScores()
            self.validation_loss = torch.nn.BCEWithLogitsLoss()

            self.test_metrics = metrics.clone(prefix="test_")
            self.test_stats = torchmetrics.classification.BinaryStatScores()
            self.test_loss = torch.nn.BCEWithLogitsLoss()
        
        def training_step(self, batch, batch_idx):
            y, y_hat = self._step(batch, batch_idx)
            step_loss = self.train_loss(y_hat, y.float().unsqueeze(1))

            y_hat = torch.sigmoid(y_hat)

            self.train_metrics(y_hat.squeeze(), y)
            self.train_stats(y_hat.squeeze(), y)
            self.log_metrics("train", self.train_metrics, step_loss, self.train_stats)
# 
            return step_loss

        def validation_step(self, batch, batch_idx):
            y, y_hat = self._step(batch, batch_idx)

            step_loss = self.validation_loss(y_hat, y.float().unsqueeze(1))

            y_hat = torch.sigmoid(y_hat)
            self.validation_metrics(y_hat.squeeze(), y)
            self.validation_stats(y_hat.squeeze(), y)
            self.log_metrics("val", self.validation_metrics, step_loss, self.validation_stats)
            
            return step_loss
        
        def test_step(self, batch, batch_idx):
            y, y_hat = self._step(batch, batch_idx)
            step_loss = self.test_loss(y_hat, y.float().unsqueeze(1))

            y_hat = torch.sigmoid(y_hat)

            self.test_metrics(y_hat.squeeze(), y)
            self.test_stats(y_hat.squeeze(), y)
            self.log_metrics("test", self.test_metrics, step_loss, self.test_stats, on_epoch=False)
            
            return step_loss

        def _step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)

            return y, y_hat
        
        def log_metrics(self, prefix, collection, loss, stats, on_epoch=True):
            self.log(f"{prefix}_loss", loss, on_step=True)

            if on_epoch:
                self.log(f"{prefix}_tp", stats.tp.int(), on_step=False, on_epoch=True)
                self.log(f"{prefix}_fp", stats.fp.int(), on_step=False, on_epoch=True)
                self.log(f"{prefix}_tn", stats.tn.int(), on_step=False, on_epoch=True)
                self.log(f"{prefix}_fn", stats.fn.int(), on_step=False, on_epoch=True)
                self.log(f"{prefix}_1", stats.tp.int() + stats.fn.int(), on_step=False, on_epoch=True)
                self.log(f"{prefix}_0", stats.tn.int() + stats.fp.int(), on_step=False, on_epoch=True)
                self.log_dict(collection, on_step=False, on_epoch=True, prog_bar=True)
            else:
                self.log(f"{prefix}_tp", stats.tp.int(), on_step=True)
                self.log(f"{prefix}_fp", stats.fp.int(), on_step=True)
                self.log(f"{prefix}_tn", stats.tn.int(), on_step=True)
                self.log(f"{prefix}_fn", stats.fn.int(), on_step=True)
                self.log(f"{prefix}_1", stats.tp.int() + stats.fn.int(), on_step=True)
                self.log(f"{prefix}_0", stats.tn.int() + stats.fp.int(), on_step=True)
                self.log_dict(collection, on_step=True, prog_bar=True)
        
        def configure_optimizers(self) -> L.pytorch.utilities.types.OptimizerLRScheduler:
            # return torch.optim.SGD(self.prepare_data(), lr=self.hparams.learning_rate, momentum=0.9)
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

def train(name: str, model: LightningModule, epochs: int, logger = None):
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15),
        ModelCheckpoint(save_top_k=1, monitor="val_BinaryAccuracy", mode="max", save_last=True)
    ]

    if logger is not None:
        trainer = L.Trainer(
            max_epochs=epochs, 
            callbacks=callbacks,
            accelerator="auto", 
            devices="auto", 
            strategy="auto", 
            profiler="simple",
            default_root_dir=f"./checkpoints/{name}",
            logger=logger,
        )
    else: 
        trainer = L.Trainer(
            max_epochs=epochs, 
            callbacks=callbacks,
            accelerator="auto", 
            devices="auto", 
            strategy="auto", 
            profiler="simple",
            default_root_dir=f"./checkpoints/{name}",
        )

    tuner = L.pytorch.tuner.Tuner(
        trainer
    )

    trainer.fit(
        model=model
    )

    trainer.test(
        ckpt_path="best"
    )

    return trainer

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
    parser.add_argument('--ignore_torch_format', '-it', help='Ignore wandb', action='store_true')
    
    parser.add_argument('--epochs', '-e', type=int, help='the amount of epochs to train', default=11) 
    parser.add_argument('--batch_size', '-b', type=int, help='the batch size to use', default=1024) # default from tuner.scale_batch_size(model)
    parser.add_argument('--learning_rate', '-l', type=float, help='the learning rate to use', default=0.001) 
    parser.add_argument('--window', '-w', type=int, help='The amount of samples per window, where 1000hz = 1 second', default=1000)
    parser.add_argument('--n_workers', '-nw', type=int, help='The amount of workers', default=8)
    parser.add_argument('--standardize', '-s', help='Standardize the data', action='store_true')

    parser.add_argument('--test', '-t', help='Train with one file to speed up training for testing purposes', action='store_true')

    args = parser.parse_args()

    assert isinstance(args.data, list)
    assert len(args.data) > 0

    if len(args.data) == 1:
        if Path(args.data[0]).exists():
            # Assume path is literal path
            participants = args.data
        else:
            # Assume path is a wildcard path
            participants = glob(args.data[0])
    elif len(args.data) > 1:
        # Assume path is a list of paths
        participants = args.data
    else:
        raise FileNotFoundError(f"No participants found on {args.data}")

    participants = [Path(p).stem for p in participants]

    if len(participants) == 0:
        raise FileNotFoundError(f"No participants found on {args.data}:", participants)
    elif len(participants) <= 3:
        raise FileNotFoundError(f"Too few participants found on {args.data}:", participants)

    if args.test_size.is_integer() and args.val_size.is_integer():
        train_participants = participants[:-int(args.test_size)]
        val_participants = participants[-int(args.test_size):]
        test_participants = participants[-int(args.val_size):]
    else: 
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
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.n_workers,
        ignore_torch_format=args.ignore_torch_format,
        train_participants=train_participants if not args.test else train_participants[:10],
        val_participants=val_participants if not args.test else val_participants[:10],
        test_participants=test_participants if not args.test else test_participants[:10],
        with_standardisation=args.standardize,
        dataset_kwargs={ 'window': args.window }
    )

    logger = None
    if args.ignore_wandb == False:
        # wandb.init(
        #     project=args.project,
        #     config={
        #         "learning_rate": args.learning_rate,
        #         "batch_size": args.batch_size,
        #         "num_workers": args.n_workers,
        #         "epochs": args.epochs,
        #         "architecture": model_name,
        #         "dataset": dataset_name
        #     }
        # )
        logger = L.pytorch.loggers.WandbLogger(
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
        epochs=args.epochs,
        logger=logger
    )

    
    # if args.ignore_wandb == False:
    #     wandb.finish()
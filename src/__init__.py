"""Top-level package for the Experiments."""

# Dependencies
import torch
import lightning as L
import datasets

# Info
__version__ = "1.0.0"

# Maintainer info
__author__ = "Alex Antonides"

# Subpackages
from .datamodules import *
from .datasets import *
from .models import *

import torchmetrics

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
    
def prepare(
    model: L.LightningModule,
) -> L.LightningModule:
    class Module(model, L.LightningModule):
        def __init__(self):
            self.save_hyperparameters()
            super().__init__()

        def setup(self, stage):
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
            self.__log_metrics("train", self.train_metrics, step_loss, self.train_stats)
# 
            return step_loss

        def validation_step(self, batch, batch_idx):
            y, y_hat = self._step(batch, batch_idx)

            step_loss = self.validation_loss(y_hat, y.float().unsqueeze(1))

            y_hat = torch.sigmoid(y_hat)
            self.validation_metrics(y_hat.squeeze(), y)
            self.validation_stats(y_hat.squeeze(), y)
            self.__log_metrics("val", self.validation_metrics, step_loss, self.validation_stats)
            
            return step_loss
        
        def test_step(self, batch, batch_idx):
            y, y_hat = self._step(batch, batch_idx)
            step_loss = self.test_loss(y_hat, y.float().unsqueeze(1))

            y_hat = torch.sigmoid(y_hat)

            self.test_metrics(y_hat.squeeze(), y)
            self.test_stats(y_hat.squeeze(), y)
            self.__log_metrics("test", self.test_metrics, step_loss, self.test_stats, on_epoch=False)
            
            return step_loss

        def _step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)

            return y, y_hat
        
        def configure_optimizers(self) -> L.pytorch.utilities.types.OptimizerLRScheduler:
            return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        
        def __log_metrics(self, prefix, collection, loss, stats, on_epoch=True):
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

    return Module()
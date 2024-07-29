import lightning as L
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset

from typing import Union

class MultiParticipantDataModule(L.LightningDataModule):
    def __init__(
        self, 
        path: str,
        train_participants: list,
        validation_participants: list,
        test_participants: list,
        batch_size: int,
        dataset: Dataset = None,
        preprocessor=None,
        standardize: bool = False
    ):
        super().__init__()
        self.path = path
        self.dataset = dataset
        
        self.train_participants = train_participants
        self.validation_participants = validation_participants
        self.test_participants = test_participants
        
        self.batch_size = batch_size

        self.preprocessor = preprocessor
        self.standardize = standardize

    def prepare_data(self):
        self.data = load_dataset(
            self.path,
            trust_remote_code=True,
            train_participants=self.train_participants, 
            val_participants=self.val_participants, 
            test_participants=self.test_participants
        )

        if self.preprocessor is not None:
            self.data = self.preprocessor(self.data)

        if self.standardize:
            self.train_mean = self.data['fit']['signal'].mean()
            self.train_std = self.data['fit']['signal'].std()

    def setup(self, stage):
        if self.dataset is not None:
            kwargs = {}
            if self.standardize:
                kwargs['train_mean'] = self.train_mean
                kwargs['train_std'] = self.train_std
            self.dataset = self.dataset(self.data[stage], **kwargs)
        else:
            self.dataset = self.data[stage]
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.hparams.num_workers, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.hparams.num_workers, pin_memory=True)
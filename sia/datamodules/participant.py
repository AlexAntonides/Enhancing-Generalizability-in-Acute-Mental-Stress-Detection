import lightning as L
from torch.utils.data import DataLoader

from datasets import load_dataset

class MultiParticipantDataModule(L.LightningDataModule):
    def __init__(
        self, 
        dataset: str,
        train_participants: list,
        validation_participants: list,
        test_participants: list,
        batch_size: int
    ):
        super().__init__()
        self.dataset = dataset
        self.train_participants = train_participants
        self.validation_participants = validation_participants
        self.test_participants = test_participants
        self.batch_size = batch_size

    def prepare_data(self):
        self.data = load_dataset(
            self.dataset,
            trust_remote_code=True,
            train_participants=self.train_participants, 
            val_participants=self.val_participants, 
            test_participants=self.test_participants
        )

    def setup(self, stage):
        self.dataset = self.data[stage]
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, pin_memory=True)

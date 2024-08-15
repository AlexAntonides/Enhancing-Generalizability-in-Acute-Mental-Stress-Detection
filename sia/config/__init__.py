"""Config module for Stress-in-Action."""

from typing import Union

import datasets
from lightning.pytorch.trainer.states import TrainerFn

class MultiParticipantConfig(datasets.BuilderConfig):
    """Datasets BuilderConfig for MultiParticipant"""
    def __init__(
        self,
        train_participants: list[Union[str, int]] = [],
        val_participants: list[Union[str, int]] = [],
        test_participants: list[Union[str, int]] = [],
        *args,
        **kwargs
    ):
        """
        Parameters
        ----------
        train_participants : list[Union[str, int]]
            List of participants to be used for training.
        val_participants : list[Union[str, int]]
            List of participants to be used for validation. 
        test_participants : list[Union[str, int]]
            List of participants to be used for testing.
        """
        super().__init__(*args, **kwargs)
        self.train_participants = train_participants
        self.val_participants = val_participants
        self.test_participants = test_participants

class MultiParticipant(datasets.ArrowBasedBuilder):
    """MultiParticipant dataset."""
    def _split_generators(self, *args, **kwargs) -> list[datasets.SplitGenerator]:
        if self.config.data_files is None:
            self.config.data_files = {}

        if len(self.config.train_participants) > 0:
            self.config.data_files[TrainerFn.FITTING.value] = [f'{participant}.csv' if participant[-4:] != '.csv' else participant for participant in self.config.train_participants]

        if len(self.config.test_participants) > 0:
            self.config.data_files[TrainerFn.TESTING.value] = [f'{participant}.csv' if participant[-4:] != '.csv' else participant for participant in self.config.test_participants]
        
        if len(self.config.val_participants) > 0:
            self.config.data_files[TrainerFn.VALIDATING.value] = [f'{participant}.csv' if participant[-4:] != '.csv' else participant for participant in self.config.val_participants]

        return super()._split_generators(*args, **kwargs)
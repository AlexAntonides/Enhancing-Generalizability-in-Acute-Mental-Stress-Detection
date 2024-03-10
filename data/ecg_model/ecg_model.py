import datasets
from datasets.packaged_modules import csv
from PIL import Image

from glob import glob
from lightning.pytorch.trainer.states import TrainerFn

from typing import Union, Dict

class EcgModelConfig(csv.CsvConfig):
    def __init__(
        self, 
        train_participants: list[Union[str, int]] = [], 
        val_participants: list[Union[str, int]] = [], 
        test_participants: list[Union[str, int]] = [],
        *args,
        **kwargs
    ):
        super().__init__(version=datasets.Version("0.0.1"), *args, **kwargs)
        self.train_participants = train_participants
        self.val_participants = val_participants
        self.test_participants = test_participants

class EcgModel(csv.Csv):
    BUILDER_CONFIG_CLASS = EcgModelConfig

    BUILDER_CONFIGS = [
        EcgModelConfig(
            name="ecg", 
            description="Ecg dataset",
            skiprows=1,
            column_names=['signal', 'label'],
        ),
    ]

    DEFAULT_CONFIG_NAME = "ecg"

    def _info(self):
        return super()._info()

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
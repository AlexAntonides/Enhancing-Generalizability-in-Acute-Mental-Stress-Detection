
from typing import Union

from datasets import Signal, Value
from datasets.packaged_modules import csv
from sia.config import MultiParticipant, MultiParticipantConfig

class SignalConfig(MultiParticipantConfig, csv.CsvConfig):
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
        MultiParticipantConfig.__init__(self, train_participants, val_participants, test_participants)
        csv.CsvConfig.__init__(self, *args, **kwargs)

class Signal(MultiParticipant, csv.Csv):
    BUILDER_CONFIG_CLASS = SignalConfig

    BUILDER_CONFIGS = [
        SignalConfig(
            name="Signal", 
            description="Signal data",
            version="1.0.0", 
            skiprows=1,
            column_names=[
                'signal',
                'category',
            ],
            Signal=Signal({
                'signal': Value('float64'),
                'category': Value('int64'),
            })
        )
    ]

    DEFAULT_CONFIG_NAME = "Signal"
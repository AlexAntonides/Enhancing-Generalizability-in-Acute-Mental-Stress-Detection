import warnings

import re
from pathlib import Path

from datasets import Dataset, IterableDataset

from typing import Callable, Union

def write_csv(path: str) -> Callable[[str, Union[Dataset, IterableDataset]], None]:
    """Save data to a CSV file.
    
    Parameters
    ----------
    path : str
        The path to the CSV file.

    Returns
    -------
    Callable[[str, Union[Dataset, IterableDataset]], None]
        A function that saves the data to a CSV file
    """
    def inner(filename: str, ds: Union[Dataset, IterableDataset]) -> None:
        warnings.filterwarnings("ignore")
        location = Path(path)

        stem = location.stem
        if stem == '*':
            filename = filename
        else:
            match = re.findall(stem, filename)
            if len(match) == 0:
                raise ValueError('No match found', stem, filename, path)
            else:
                filename = match[0]

        location.parent.mkdir(parents=True, exist_ok=True)
        ds.to_csv(location.with_stem(filename))
        warnings.filterwarnings("default")
    return inner

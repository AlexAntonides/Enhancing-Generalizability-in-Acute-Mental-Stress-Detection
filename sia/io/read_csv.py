import warnings
from glob import glob 

import datasets
from datasets import load_dataset, Dataset, IterableDataset

from typing import Union, Tuple, Callable, Iterator

def read_csv(path: str, columns: Union[None, Tuple[str]] = None) -> Callable[[Union[None, Tuple[str]]], Iterator[Union[Dataset, IterableDataset]]]:
    """Read a CSV file.
    
    Parameters
    ----------
    path : str
        The path to the CSV file.
    columns : Union[None, Tuple[str]], optional
        The columns to read, by default, None.

    Returns
    -------
    Callable[[Union[None, Tuple[str]]], Iterator[Union[Dataset, IterableDataset]]]
        A function that reads the CSV file.
    """
    def inner() -> Iterator[Union[Dataset, IterableDataset]]:
        warnings.filterwarnings("ignore")
        files = glob(path)
        if len(files) == 0:
            raise FileNotFoundError(f'No files found at {path}')
        elif len(files) != 1:
            for file in files:
                yield from read_csv(file, columns)()
        else:
            _path = files[0]
            ds = load_dataset(
                "csv", 
                data_files=_path,
                usecols=columns
            )
            yield _path, ds['train']
        warnings.filterwarnings("default")
    return inner

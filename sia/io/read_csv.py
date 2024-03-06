from glob import glob 

import datasets
from datasets import load_dataset, Dataset, IterableDataset

from typing import Union, Tuple, Callable, Iterator

def read_csv(path: str, columns: Union[None, Tuple[str]] = None) -> Callable[[Union[None, Tuple[str]]], Iterator[Union[Dataset, IterableDataset]]]:
    """Read a CSV file."""
    def inner() -> Iterator[Union[Dataset, IterableDataset]]:
        files = glob(path)
        if len(files) == 0:
            raise FileNotFoundError(f'No files found at {path}')
        elif len(files) != 1:
            for file in files:
                yield from read_csv(file, columns)()
        else:
            ds = load_dataset(
                "csv", 
                data_files=path,
                usecols=columns, 
                features=datasets.Features({
                    columns[0]: datasets.Value('float64'), 
                    columns[1]: datasets.Value('string')
                }) if columns is not None else None
            )
            yield path, ds['train']
    return inner

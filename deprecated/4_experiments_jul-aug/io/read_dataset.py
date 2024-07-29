from glob import glob 
from pathlib import Path

import datasets
from datasets import load_dataset, Dataset, IterableDataset

from typing import Union, Tuple, Callable, Iterator

def read_dataset(path: str) -> Callable[[Union[None, Tuple[str]]], Iterator[Union[Dataset, IterableDataset]]]:
    """Read a CSV file."""
    def inner() -> Iterator[Union[Dataset, IterableDataset]]:
        files = glob(path)
        if len(files) == 0:
            raise FileNotFoundError(f'No files found at {path}')
        elif len(files) != 1:
            for file in files:
                yield from read_dataset(file)()
        else:
            ds = load_dataset(
                Path(path).parent,
                files=path,
                trust_remote_code=True,
                num_proc=8 if len(files) > 8 else len(files)
            )
            yield path, ds['train']
    return inner

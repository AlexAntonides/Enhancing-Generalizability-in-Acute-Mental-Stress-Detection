from glob import glob 

import pandas as pd 
import numpy as np

from typing import Union, Tuple, Callable, Iterator

def read_csv(path: str) -> Callable[[Union[None, Tuple[str]]], Iterator[Union[str, np.ndarray]]]:
    """Read a CSV file."""
    def inner(header: Union[None, Tuple[str]] = None) -> Iterator[Union[str, np.ndarray]]:
        files = glob(path)
        if len(files) == 0:
            raise FileNotFoundError(f'No files found at {path}')
        elif len(files) != 1:
            for file in files:
                yield from read_csv(file)()
        else:
            df = pd.read_csv(files[0], header=header, low_memory=False)
            yield files[0], df.to_numpy()
    return inner

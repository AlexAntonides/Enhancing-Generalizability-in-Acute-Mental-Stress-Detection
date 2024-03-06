import re
from pathlib import Path

from datasets import Dataset, IterableDataset

from typing import Callable, Union

def write_png(directory: str) -> Callable[[str, Union[Dataset, IterableDataset]], None]:
    """Save data to a PNG file."""
    def inner(subdirectory: str, ds: Union[Dataset, IterableDataset]) -> None:
        location = Path(directory)

        stem = location.stem
        if stem == '*':
            subdirectory = subdirectory
        else:
            match = re.findall(stem, subdirectory)
            if len(match) == 0:
                raise ValueError('No match found', stem, subdirectory, directory)
            else:
                subdirectory = match[0]

        location.parent.mkdir(parents=True, exist_ok=True)
        raise Exception("Doesn't work fully yet")
        ds.save_to_disk(location.with_stem(subdirectory))
    return inner
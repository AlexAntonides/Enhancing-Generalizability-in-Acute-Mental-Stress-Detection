from glob import glob 

import pandas as pd 
import numpy as np

import mne

from datasets import Dataset, IterableDataset

from typing import Union, Callable, Iterator, Tuple

from .metadata import Metadata

def read_edf(path: str, metadata: Union[pd.DataFrame, Metadata, Callable[[mne.io.Raw], pd.DataFrame]] = None, sampling_rate: int = 1000) -> Callable[[], Tuple[str, Union[Dataset, IterableDataset]]]:
    """Read an EDF file.
    
    Parameters
    ----------
    path : str
        The path to the EDF file.
    metadata : Union[pd.DataFrame, Metadata, Callable[[mne.io.Raw], pd.DataFrame]], optional
        The metadata for the EDF file, by default, None.
    sampling_rate : int, optional
        The sampling rate of the EDF file, by default, 1000.

    Returns
    -------
    Callable[[], Tuple[str, Union[Dataset, IterableDataset]]]
        A function that reads the EDF file.
    """
    def inner() -> Iterator[Tuple[str, Union[Dataset, IterableDataset]]]:
        files = glob(path)
        if len(files) == 0:
            raise FileNotFoundError(f'No files found at {path}')
        elif len(files) != 1:
            if isinstance(metadata, pd.DataFrame):
                raise ValueError('Metadata cannot be provided for multiple files')
            else:
                for file in files:
                    yield from read_edf(file, metadata)()
        else:
            _path = files[0]

            raw = mne.io.read_raw_edf(files[0], preload=True, verbose=40) # 40 = Logging.ERROR
            timestamps = raw.times * sampling_rate # The code will interpret this value as milliseconds, thus 1 millisecond, instead of 0.001 millisecond. 
            signals = raw.get_data()[0]

            timestamps = timestamps.astype('timedelta64[ms]') + np.datetime64(raw.annotations.orig_time.replace(tzinfo=None), 'ms')

            df = pd.DataFrame({'timestamp': timestamps, 'signal': signals})

            if isinstance(metadata, pd.DataFrame):
                df = attach_edf_metadata(df, metadata)
            elif isinstance(metadata, Metadata):
                df = attach_edf_metadata(df, metadata.retrieve(_path))
            elif callable(metadata):
                df = attach_edf_metadata(df, metadata(raw))

            yield _path, Dataset.from_pandas(df)
    return inner

def attach_edf_metadata(df: pd.DataFrame, metadata: Metadata) -> pd.DataFrame: 
    """Attach metadata to an EDF file.
    
    Parameters
    ----------
    df : pd.DataFrame
        The EDF file.
    metadata : Metadata
        The metadata for the EDF file.
    
    Returns
    -------
    pd.DataFrame
        The EDF file with the metadata attached
    """
    if not 'datetime64[ms]' in str(metadata.start.dtype) or not 'datetime64[ms]' in str(metadata.end.dtype):
        with pd.option_context('mode.chained_assignment', None):
            metadata.start = metadata.start.astype('datetime64[ms]')
            metadata.end = metadata.end.astype('datetime64[ms]')

    df = pd.merge_asof(df.sort_values('timestamp'), pd.DataFrame(metadata, columns=['category', 'start', 'end']), left_on='timestamp', right_on='start', direction='backward')
    df = df.drop(columns=['start', 'end'])

    return df
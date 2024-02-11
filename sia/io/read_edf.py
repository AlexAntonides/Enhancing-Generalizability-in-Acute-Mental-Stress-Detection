from glob import glob 

import pandas as pd 
import numpy as np

import mne

from typing import Union, Callable, Generator

from .metadata import Metadata

def read_edf(path: str, metadata: Union[Metadata, Callable[[mne.io.Raw], pd.DataFrame]] = None) -> Generator[pd.DataFrame, None, None]:
    """Read an EDF file."""

    files = glob(path)
    if len(files) == 0:
        raise FileNotFoundError(f'No files found at {path}')
    elif len(files) != 1:
        if isinstance(metadata, pd.DataFrame):
            raise ValueError('Metadata cannot be provided for multiple files')
        else:
            for file in files:
                yield from read_edf(file, metadata)
    else:
        raw = mne.io.read_raw_edf(path, preload=True, verbose=40) # 40 = Logging.ERROR
        timestamps = raw.times * 1000 # The code will interpret this value as milliseconds, thus 1 millisecond, instead of 0.001 millisecond. 
        signals = raw.get_data()[0]

        timestamps = timestamps.astype('timedelta64[ms]') + np.datetime64(raw.annotations.orig_time.replace(tzinfo=None), 'ms')

        df = pd.DataFrame({'timestamp': timestamps, 'signal': signals})
        if isinstance(metadata, pd.DataFrame):
            df = attach_edf_metadata(df, metadata)
        elif isinstance(metadata, Metadata):
            df = attach_edf_metadata(df, metadata.retrieve(path))
        elif callable(metadata):
            df = attach_edf_metadata(df, metadata(raw))

        yield df

def attach_edf_metadata(df: pd.DataFrame, metadata: Metadata) -> pd.DataFrame: 
    """Attach metadata to an EDF file."""
    if not 'datetime64[ms]' in str(metadata.start.dtype) or not 'datetime64[ms]' in str(metadata.end.dtype):
        with pd.option_context('mode.chained_assignment', None):
            metadata.start= metadata.start.astype('datetime64[ms]')
            metadata.end = metadata.end.astype('datetime64[ms]')

    df = pd.merge_asof(df.sort_values('timestamp'), pd.DataFrame(metadata, columns=['category', 'start', 'end']), left_on='timestamp', right_on='start', direction='backward')
    df = df.drop(columns=['start', 'end'])

    return df
import sys
from typing import Any, TypeVar, Iterator, Union, Callable

from pathlib import Path
from prettytable import PrettyTable

T = TypeVar('T')
if sys.version_info >= (3,11):
    from typing import Self 
else:
    Self = Any

import numpy as np
from datasets import Dataset, IterableDataset

from .pipeline import Pipeline

class Preprocessing(Pipeline):
    """
    The preprocessing pipeline class in the SiA-Kit to preprocess datasets.
    """
    __READER_SETTING_KEY = 'reader'
    __PROCESS_SETTING_KEY = 'process'
    __FILTER_SETTING_KEY = 'filter'
    __SELECT_SETTING_KEY = 'select'
    __DROP_SETTING_KEY = 'drop'
    __RENAME_SETTING_KEY = 'rename'

    def __init__(self):
        super().__init__()

        self.__path = None
        self.__dataset = None

    ## --- Pipeline Configurators --- ##
    def data(self, reader: Callable[[str], Iterator[np.ndarray]]) -> Self:
        """Configuration to read data using a given reader."""
        return self._set(self.__READER_SETTING_KEY, reader)
    
    def process(self, processor: Callable[[Union[Dataset, IterableDataset]], Union[Dataset, IterableDataset]]) -> Self:
        """Configuration to process data using a given processor."""
        return self._set(self.__PROCESS_SETTING_KEY, processor)
    
    def filter(self, filter: Callable[[Union[Dataset, IterableDataset]], Union[Dataset, IterableDataset]]) -> Self:
        """Configuration to filter data using a given filter."""
        return self._set(self.__FILTER_SETTING_KEY, filter)
    
    def select(self, columns: Union[str, list[str]]) -> Self:
        """Configuration to select columns from the dataset."""
        setting = []
        if isinstance(columns, str):
            setting.append(columns)
        else: 
            setting.extend(columns)
        return self._set(self.__SELECT_SETTING_KEY, setting)
    
    def drop(self, columns: Union[str, list[str]]) -> Self:
        """Configuration to drop columns from the dataset."""
        setting = []
        if isinstance(columns, str):
            setting.append(columns)
        else: 
            setting.extend(columns)
        return self._set(self.__DROP_SETTING_KEY, setting)
    
    def rename(self, x: Union[str, dict, list[str, dict]], y: str = None) -> Self:
        """Configuration to rename columns in the dataset."""
        setting = []
        def _rename(x: Union[str, dict, list[str, dict]], y: str = None) -> list[dict]:
            if isinstance(x, str):
                return [{x: y}]
            elif isinstance(x, dict):
                return [x]
            elif isinstance(x, list):
                if len(x) == 2 and isinstance(x[0], str):
                    return _rename(x[0], x[1])
                elif isinstance(x[0], dict):
                    return _rename(x)
                else: 
                    raise ValueError('Invalid input')
            else: 
                raise ValueError('Invalid input')

        setting.extend(_rename(x, y))
        return self._set(self.__RENAME_SETTING_KEY, setting)
    
    ## --- Pipeline Executors --- ##
    def to(self, writer: Callable[[Union[Dataset, IterableDataset]], Union[Dataset, IterableDataset]]) -> None:
        """Execute the pipeline and write the dataset using a given writer."""
        for _ in self:
            filename = Path(self.__path).stem
            writer(filename, self.__dataset)

    def __next__(self):
        """Gets the next setting in the pipeline settings, and execute it."""
        key, value = super().__next__()
        self.__execute(key, value)
        return key, value

    def __execute(self, key: str, value: Union[Any, list]) -> Iterator[np.ndarray]:
        """Executes the given key-value pair in the pipeline settings."""
        def _get_args(fn):
            """Get the arguments of a given function"""
            return fn.__code__.co_varnames[:fn.__code__.co_argcount]
        
        if isinstance(value, list):
            for _ in value:
                self.__execute(key, _)
        else:
            if key == self.__READER_SETTING_KEY:
                for path, dataset in value():
                    self.__dataset = dataset
                    self.__path = path
            elif key == self.__PROCESS_SETTING_KEY:
                    args = _get_args(value)
                    features = self.__dataset.features.keys()
                    intersection = list(set(features) & set(args))

                    kwargs = {}
                    if 'dataset' in args:
                        kwargs['dataset'] = self.__dataset

                    self.__dataset = self.__dataset.map(
                        value,
                        batched=True,
                        batch_size=int(len(self.__dataset) * .25),
                        input_columns=intersection,
                        with_indices='idxs' in args,
                        fn_kwargs=kwargs
                    )
            elif key == self.__SELECT_SETTING_KEY:
                self.__dataset = self.__dataset.select_columns(value)
            elif key == self.__DROP_SETTING_KEY:
                self.__dataset = self.__dataset.remove_columns(value)
            elif key == self.__RENAME_SETTING_KEY:
                for column in value:
                    self.__dataset = self.__dataset.rename_columns(column)
            elif key == self.__FILTER_SETTING_KEY:
                args = _get_args(value)
                features = self.__dataset.features.keys()
                intersection = list(set(features) & set(args))

                kwargs = {}
                if 'dataset' in args:
                    kwargs['dataset'] = self.__dataset

                self.__dataset = self.__dataset.filter(
                    value,
                    batched=True,
                    batch_size=int(len(self.__dataset) * .25),
                    input_columns=intersection,
                    with_indices='idxs' in args,
                    fn_kwargs=kwargs,
                    num_proc=8
                )
            else:
                raise ValueError(f"Invalid key: {key}")
            
    def __repr__(self, table=PrettyTable()):
        """Generate a representation of the Preprocessing pipeline settings."""
        table.add_row(['Path', self.__path])
        table = super().__repr__(table)
        return table 
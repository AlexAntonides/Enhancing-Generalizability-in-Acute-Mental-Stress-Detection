import warnings 
warnings.filterwarnings('ignore')

from pathlib import Path

from tqdm import tqdm
from tabulate import tabulate

import numpy as np
import pandas as pd

import inspect
from glob import glob

from joblib import Parallel, delayed
from ..utils import tqdm_joblib

import datasets
from datasets import Dataset, IterableDataset

from typing import Iterator, Union, Callable

class Pipeline:
    def __init__(self):
        self.settings = []

    def data(self, reader: Callable[[str], Iterator[np.ndarray]]):
        self.settings.append({'reader': reader})
        return self
    
    def process(self, processor: Callable[[Union[Dataset, IterableDataset]], Union[Dataset, IterableDataset]]):
        self.settings.append({'process': processor})
        return self
    
    def select(self, columns: Union[str, list[str]]):
        setting = []
        if isinstance(columns, str):
            setting.append(columns)
        else: 
            setting.extend(columns)
        self.settings.append({'select': setting})
        return self
    
    def drop(self, columns: Union[str, list[str]]):
        setting = []
        if isinstance(columns, str):
            setting.append(columns)
        else: 
            setting.extend(columns)
        self.settings.append({'drop': setting})
        return self
    
    def rename(self, x: Union[str, dict, list[str, dict]], y: str = None):
        setting = []
        def _rename(x: Union[str, dict, list[str, dict]], y: str = None):
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
        self.settings.append({'rename': setting})
        return self
    
    def read(self) -> Union[np.ndarray]:
        result = []
        for _, dataset in self._iterate():
            result.append(dataset)
        return datasets.concatenate_datasets(result)    
    
    def to(self, writer: Callable[[Union[Dataset, IterableDataset]], Union[Dataset, IterableDataset]]):
        for path, dataset in self._iterate():
            filename = Path(path).stem
            writer(filename, dataset)

    def iterate(self) -> Iterator[np.ndarray]:
        for _, dataset in self._iterate():
            yield dataset

    def _iterate(self) -> Iterator[np.ndarray]:
        if 'reader' not in list(self.settings[0].keys())[0]:
            raise ValueError("Can't apply functions without data. Please ensure the first method called reads data.")

        for index, setting in enumerate(self.settings):
            for key, value in setting.items():
                if key == 'reader':
                    # variables = inspect.getclosurevars(value).nonlocals
                    # if 'path' in variables:
                    #     n = len(glob(variables['path']))
                    for path, dataset in value():
                        if len(self.settings[index+1:]) > 0:
                            dataset = self._execute(dataset, self.settings[index+1:])
                            
                        yield path, dataset
                else: 
                    continue


    def _execute(self, dataset, setting: Union[dict, list]) -> Iterator[np.ndarray]:
        def _get_args(fn):
            return fn.__code__.co_varnames[:fn.__code__.co_argcount]
        
        if isinstance(setting, list):
            for x in setting:
                dataset = self._execute(dataset, x)
        else:
            for key, value in setting.items():
                if key == 'process':
                    args = _get_args(value)
                    features = dataset.features.keys()
                    intersection = list(set(features) & set(args))

                    kwargs = {}
                    if 'dataset' in args:
                        kwargs['dataset'] = dataset

                    dataset = dataset.map(
                        value,
                        batched=True,
                        batch_size=int(len(dataset) * .25),
                        input_columns=intersection,
                        with_indices='idxs' in args,
                        fn_kwargs=kwargs
                    )
                elif key == 'select':
                    dataset = dataset.select_columns(value)
                elif key == 'drop':
                    dataset = dataset.remove_columns(value)
                elif key == 'rename':
                    for column in value:
                        dataset = dataset.rename_columns(column)
                else:
                    break
                
        return dataset

    def __repr__(self):
        data = self.read()
        return tabulate([
            ['n', sum([len(d) for d in data])],
            ['features', *data[0].column_names],
        ], tablefmt='fancy_grid')
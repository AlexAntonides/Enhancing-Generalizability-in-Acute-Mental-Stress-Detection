from pathlib import Path

from tqdm import tqdm

import numpy as np
import pandas as pd

import inspect
from glob import glob

from joblib import Parallel, delayed
from ..utils import tqdm_joblib

from typing import Self, Iterator, Union, Callable

class Pipeline:
    def __init__(self):
        self.settings = {}

    def data(self, reader: Callable[[str], Iterator[np.ndarray]]) -> Self:
        self.settings['reader'] = reader
        return self
    
    def preprocess(self, preprocessor: Callable[[np.ndarray], pd.DataFrame]) -> Self:
        if 'preprocess' not in self.settings:
            self.settings['preprocess'] = []
        self.settings['preprocess'].append(preprocessor)
        return self
    
    def postprocess(self, postprocessor: Callable[[np.ndarray], pd.DataFrame]) -> Self:
        if 'postprocess' not in self.settings:
            self.settings['postprocess'] = []
        self.settings['postprocess'].append(postprocessor)
        return self
    
    def reduce(self, reducer: Callable[[np.ndarray], pd.DataFrame]) -> Self:
        if 'reduce' not in self.settings:
            self.settings['reduce'] = []
        self.settings['reduce'].append(reducer)
        return self
    
    def to(self, writer: Callable[[np.ndarray], pd.DataFrame], n_jobs=4):
        for path, data in self._iterate(n_jobs):
            filename = Path(path).stem
            writer(filename, data)

    def read(self, n_jobs=4) -> Union[np.ndarray]:
        result = []
        for _, data in self._iterate(n_jobs):
            result.append(data)
        result = np.concatenate(result)
            
        return result
    
    def iterate(self, n_jobs=4) -> Iterator[np.ndarray]:
        for _, data in self._iterate(n_jobs):
            yield data

    def _iterate(self, n_jobs=4) -> Iterator[np.ndarray]:
        if 'reader' not in self.settings:
            raise ValueError('No reader has been set.')

        variables = inspect.getclosurevars(self.settings['reader']).nonlocals
        if 'path' in variables:
            n = len(glob(variables['path']))
        
        with tqdm_joblib(tqdm(total=n)):
            Parallel(n_jobs=n_jobs)(delayed(self._execute)(path, data) for path, data in self.settings['reader']())
            
    def _execute(self, path: str, data: np.ndarray) -> np.ndarray:
        if 'preprocess' in self.settings:
            for preprocessor in self.settings['preprocess']:
                data = preprocessor(data)
            
        if 'reduce' in self.settings:
            for reducer in self.settings['reduce']:
                data = reducer(data)

        if 'postprocess' in self.settings:
            for postprocesser in self.settings['postprocess']:
                data = postprocesser(data)

        return path, data
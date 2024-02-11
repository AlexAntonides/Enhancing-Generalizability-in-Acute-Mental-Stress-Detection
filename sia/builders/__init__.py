"""Builders submodule for Stress-in-Action."""
import pathlib

import pandas as pd

from ..io import Metadata, read_edf

from typing import Self, Union, Tuple

class Pipeline:
    def __init__(self):
        self.raw_path = None
        self.meta = None

    def raw(self, path: str) -> Self:
        self.raw_path = path
        return self

    def metadata(self, metadata: Metadata) -> Self:
        self.meta = metadata
        return self
    
    def execute(self) -> Union[pd.DataFrame, Tuple[pd.DataFrame]]:
        result = None
        if pathlib.Path(self.raw_path).suffix == '.edf':
            result = []
            for df in read_edf(self.raw_path, self.meta):
                result.append(df)
            result = pd.concat(result)
            
        return result
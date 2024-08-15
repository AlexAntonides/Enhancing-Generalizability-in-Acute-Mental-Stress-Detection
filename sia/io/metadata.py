import sys
import re

import pandas as pd

from typing import Any, Pattern
if sys.version_info >= (3,11):
    from typing import Self 
else:
    Self = Any

class Metadata:
    """Reads metadata from a file."""
    def __init__(self, path: str):
        """
        Parameters
        ----------
        path : str
            The path to the metadata file.
        """
        self.path = path
        self.regex = None

    def on_regex(self, regex: Pattern[str]) -> Self:
        """Set a regex to extract the subject ID from the path.
        
        Parameters
        ----------
        regex : Pattern[str]
            The regex pattern.
        
        Returns
        -------
        Self
            The instance of the class for method chaining.
        """
        self.regex = regex
        return self

    def retrieve(
        self, 
        path: str,
        names: list[str] = ['subject_id', 'category', 'code', 'start', 'end'],
        dtype: dict[str, str] = {'subject_id': 'str', 'category': 'str', 'code': 'str', 'start': 'str', 'end': 'str'},
        subject_column: str = 'subject_id'
    ):
        """Retrieve the metadata for a specific subject.

        Parameters
        ----------
        path : str
            The path to the ECG signal file.
        names : list[str], optional
            The names of the columns in the metadata file, by default, SiA uses ['subject_id', 'category', 'code', 'start', 'end'].
        dtype : dict[str, str], optional
            The data types of the columns in the metadata file, by default, SiA uses {'subject_id': 'str', 'category': 'str', 'code': 'str', 'start': 'str', 'end': 'str'}
        subject_column : str, optional
            The column name that contains the subject ID, by default, SiA uses 'subject_id'.
        
        Returns
        -------
        pd.DataFrame
            The metadata for the subject.
        """
        df = pd.read_csv(
            self.path, 
            sep='\t', 
            decimal=',', 
            skiprows=[0], 
            header=None,
            names=names,
            dtype=dtype
        )

        if self.regex is not None:
            match = re.findall(self.regex, path)
            if len(match) == 0:
                raise ValueError('No match found')
            else:
                return df[df[subject_column] == match[0]]
        else:
            raise ValueError('No valid parser provided')
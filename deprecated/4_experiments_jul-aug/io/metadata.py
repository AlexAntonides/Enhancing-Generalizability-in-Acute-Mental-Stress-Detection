import re

import pandas as pd

from typing import Pattern

class Metadata:
    """Reads metadata from a file."""
    def __init__(self, path: str):
        self.path = path
        self.regex = None

    def on_regex(self, regex: Pattern[str]):
        self.regex = regex
        return self

    def retrieve(self, path: str):
        # This is kind of hardcoded now. :(
        df = pd.read_csv(
            self.path, 
            sep='\t', 
            decimal=',', 
            skiprows=[0], 
            header=None, 
            names=['subject_id', 'category', 'code', 'start', 'end'], 
            dtype={'subject_id': 'str', 'category': 'str', 'code': 'str', 'start': 'str', 'end': 'str'}, 
        )

        if self.regex is not None:
            match = re.findall(self.regex, path)
            if len(match) == 0:
                raise ValueError('No match found')
            else:
                return df[df['subject_id'] == match[0]]
        else:
            raise ValueError('No valid parser provided')
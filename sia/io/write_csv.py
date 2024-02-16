import re
from pathlib import Path

import pandas as pd
import numpy as np

def write_csv(path: str):
    """Save data to a CSV file."""
    def inner(filename: str, data: np.ndarray) -> None:
        location = Path(path)

        stem = location.stem
        if stem == '*':
            filename = filename
        else:
            match = re.findall(stem, filename)
            if len(match) == 0:
                raise ValueError('No match found', stem, filename, path)
            else:
                filename = match[0]

        location.parent.mkdir(parents=True, exist_ok=True)
        # np.savetxt(location.with_stem(filename), data, delimiter=',')
        pd.DataFrame(data).to_csv(location.with_stem(filename), index=False, header=False)
    return inner
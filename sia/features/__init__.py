"""Features module for Stress-in-Action."""

# Aliases
from .time_domain import *
from .frequency_domain import *
from .nonlinear_domain import *
from .morphology_domain import *

# Packages
from enum import Enum
from typing import Union

import numpy as np
import neurokit2 as nk

# Helpers
def extract_peaks(peaks: list[int]) -> list[int]:
    """Extract the peak indices from the list of peaks, where peaks are non-zero values.

    Parameters
    ----------
    peaks : list[int]
        The list of peaks.

    Returns
    -------
    np.ndarray
        An array with the indices of the peaks
    """
    return np.nonzero(peaks)[0]

class Waves(str, Enum):
    T_Peak = "ECG_T_Peaks"

def delineate(features: Union[Waves, list[Waves]]):
    """Compute ECG delineation features.

    Parameters
    ----------
    features : Union[Waves, list[Waves]]
        The features to be computed.

    Returns
    -------
    function
        A function that computes the ECG delineation features.
    """
    def inner(ECG_Clean: list[float]):
        return nk.ecg_delineate(ECG_Clean, method="peaks")[0][features]
    return inner
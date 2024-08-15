import warnings
from warnings import warn

try:
    import cupy as cp
    np = cp
except ImportError:
    import numpy as np
    
from enum import Enum

class Feature(str, Enum):
    TWA = "twa"
    """T-wave alternans (TWA) feature."""

def morphology_domain(features: tuple[Feature]):
    """Compute morphology domain features.

    Parameters
    ----------
    features : tuple[Feature]
        A tuple with the features to be computed.

    Returns
    -------
    function
        A function that computes the features in the morphology domain.    
    """
    def inner(ECG_Clean: list[float], tpeaks: list[int]):
        result = {}
        warnings.filterwarnings("ignore")
        for feature in features:
            if feature == Feature.TWA:
                twa = calculate_twa(ECG_Clean, tpeaks)
                result.update({ "twa": twa })
            else:
                raise ValueError(f"Feature {feature} is not valid.")
        warnings.filterwarnings("default")
        return result
    return inner

def calculate_twa(signal: list[float], tpeaks: list[int]):
    """Compute the T-wave alternans (TWA) feature.

    Parameters
    ----------
    signal : list[float]
        The ECG signal.
    tpeaks : list[int]
        The T-wave peaks.
        
    Returns
    -------
    dict
        A dictionary containing the TWA feature.
    """
    # Divide the T-peaks into two buckets, even and odd.
    even_bucket = tpeaks[1::2]
    odd_bucket = tpeaks[::2]

    # Calculate the average of the even and odd buckets.
    average_t_even = np.mean(np.take(signal, even_bucket))
    average_t_odd = np.mean(np.take(signal, odd_bucket))

    if average_t_even is None or average_t_odd is None:
        return np.nan.item()
    else:
        # Calculate the difference in amplitude between the even and odd buckets.
        twa = abs(average_t_even - average_t_odd)
        return twa.item()
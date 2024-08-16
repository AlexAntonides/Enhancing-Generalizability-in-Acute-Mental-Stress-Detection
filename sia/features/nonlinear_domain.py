import warnings

import pandas as pd 
import neurokit2 as nk 

try:
    import cupy as cp
    np = cp
except ImportError:
    import numpy as np
    
from enum import Enum

class Feature(str, Enum):
    ENTROPY = "entropy"
    """Entropy features."""
    POINCARE = "poincare"
    """Poincaré features."""
    FRAGMENTATION = "fragmentation"
    """Indices of Heart Rate Fragmentation (Costa, 2017)"""
    RQA = "rqa"
    """Recurrence Quantification Analysis (RQA) features."""

def nonlinear_domain(features: tuple[Feature], sampling_rate: int = 1000):
    """Compute nonlinear domain features.

    Parameters
    ----------
    features : tuple[Feature]
        A tuple with the features to be computed.
    sampling_rate : int
        The sampling rate of the ECG signal.

    Returns
    -------
    function
        A function that computes the features in the nonlinear domain
    """
    def inner(rpeaks: list[int]):
        result = {}
        warnings.filterwarnings("ignore")
        for feature in features:
            if feature == Feature.ENTROPY:
                hrv_nonlinear = nk.hrv_nonlinear(rpeaks, sampling_rate=sampling_rate)
                hrv_nonlinear = hrv_nonlinear.fillna(0)
                hrv_nonlinear = hrv_nonlinear.iloc[0].to_dict()

                result.update({
                    f'apen': hrv_nonlinear.get("HRV_ApEn", 0),
                    f'sampen': hrv_nonlinear.get("HRV_SampEn", 0),
                    f'fuzzyen': hrv_nonlinear.get("HRV_FuzzyEn", 0),
                })
            elif feature == Feature.POINCARE:
                hrv_nonlinear = nk.hrv_nonlinear(rpeaks, sampling_rate=sampling_rate)
                hrv_nonlinear = hrv_nonlinear.fillna(0)
                hrv_nonlinear = hrv_nonlinear.iloc[0].to_dict()

                result.update({
                    f'sd1': hrv_nonlinear.get("HRV_SD1", 0),
                    f'sd2': hrv_nonlinear.get("HRV_SD2", 0),
                    f'sd1_sd2': hrv_nonlinear.get("HRV_SD1/SD2", 0),
                })
            elif feature == Feature.FRAGMENTATION:
                hrv_nonlinear = nk.hrv_nonlinear(rpeaks, sampling_rate=sampling_rate)
                hrv_nonlinear = hrv_nonlinear.fillna(0)
                hrv_nonlinear = hrv_nonlinear.iloc[0].to_dict()

                result.update({
                    f'pss': hrv_nonlinear.get("HRV_PSS", 0),
                })
            elif feature == Feature.RQA:
                rqa = nk.hrv_rqa(rpeaks, sampling_rate=sampling_rate)
                rqa = rqa.fillna(0)

                result.update({
                    f"w": rqa.get('W', 0),
                    f"wmax": rqa.get('WMax', 0),
                    f"wen": rqa.get('WEn', 0)
                })
            else:
                raise ValueError(f"Feature {feature} is not valid.")
        warnings.filterwarnings("default")
        return result
    return inner
    
import numpy as np
import pandas as pd

import neurokit2 as nk

from typing import Union

def pantompkins(signal: np.ndarray, sampling_rate: int = 1000) -> pd.DataFrame:
    return nk.ecg_process(signal, sampling_rate=sampling_rate, method='pantompkins1985')

def hamilton(signal: np.ndarray, sampling_rate: int = 1000) -> pd.DataFrame:
    return nk.ecg_process(signal, sampling_rate=sampling_rate, method='hamilton2002')

def elgendi(signal: np.ndarray, sampling_rate: int = 1000) -> pd.DataFrame:
    return nk.ecg_process(signal, sampling_rate=sampling_rate, method='elgendi2010')

def engzeemod(signal: np.ndarray, sampling_rate: int = 1000) -> pd.DataFrame:
    return nk.ecg_process(signal, sampling_rate=sampling_rate, method='engzeemod2012')

def neurokit(signal: np.ndarray, sampling_rate: int = 1000) -> pd.DataFrame:
    return nk.ecg_process(signal, sampling_rate=sampling_rate, method='neurokit')
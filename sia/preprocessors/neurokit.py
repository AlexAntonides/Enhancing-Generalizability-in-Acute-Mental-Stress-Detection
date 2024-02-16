import numpy as np
import pandas as pd

import neurokit2 as nk

from typing import Tuple, Callable

def pantompkins(sampling_rate: int = 1000, use_cols: Tuple[str] = ['ECG_Clean']) -> Callable[[np.ndarray], np.ndarray]:
    def inner(data: np.ndarray): 
        # The ith column is hard-coded :(
        df, _ = nk.ecg_process(list(data[:,1]), sampling_rate=sampling_rate, method='pantompkins1985')
        data = np.concatenate([np.delete(data, 1, axis=1), df[use_cols].to_numpy()], axis=1)
        return data
    return inner

def hamilton(sampling_rate: int = 1000, use_cols: Tuple[str] = ['ECG_Clean']) -> Callable[[np.ndarray], np.ndarray]:
    def inner(data: np.ndarray): 
        # The ith column is hard-coded :(
        df, _ = nk.ecg_process(list(data[:,1]), sampling_rate=sampling_rate, method='hamilton2002')
        data = np.concatenate([np.delete(data, 1, axis=1), df[use_cols].to_numpy()], axis=1)
        return data
    return inner

def elgendi(sampling_rate: int = 1000, use_cols: Tuple[str] = ['ECG_Clean']) -> Callable[[np.ndarray], np.ndarray]:
    def inner(data: np.ndarray): 
        # The ith column is hard-coded :(
        df, _ = nk.ecg_process(list(data[:,1]), sampling_rate=sampling_rate, method='elgendi2010')
        data = np.concatenate([np.delete(data, 1, axis=1), df[use_cols].to_numpy()], axis=1)
        return data
    return inner

def engzeemod(sampling_rate: int = 1000, use_cols: Tuple[str] = ['ECG_Clean']) -> Callable[[np.ndarray], np.ndarray]:
    def inner(data: np.ndarray): 
        # The ith column is hard-coded :(
        df, _ = nk.ecg_process(list(data[:,1]), sampling_rate=sampling_rate, method='engzeemod2012')
        data = np.concatenate([np.delete(data, 1, axis=1), df[use_cols].to_numpy()], axis=1)
        return data
    return inner

def neurokit(sampling_rate: int = 1000, use_cols: Tuple[str] = ['ECG_Clean']) -> Callable[[np.ndarray], np.ndarray]:
    def inner(data: np.ndarray): 
        # The ith column is hard-coded :(
        df, _ = nk.ecg_process(list(data[:,1]), sampling_rate=sampling_rate, method='neurokit')
        data = np.concatenate([np.delete(data, 1, axis=1), df[use_cols].to_numpy()], axis=1)
        return data
    return inner
import numpy as np
import pandas as pd

import neurokit2 as nk

from typing import Callable

def pantompkins(sampling_rate: int = 1000) -> Callable[[list], dict]:
    def inner(signal): 
        df, _ = nk.ecg_process(signal, sampling_rate=sampling_rate, method='pantompkins1985')
        return df.to_dict('list')
    return inner

def hamilton(sampling_rate: int = 1000) -> Callable[[list], dict]:
    def inner(signal): 
        df, _ = nk.ecg_process(signal, sampling_rate=sampling_rate, method='hamilton2002')
        return df.to_dict('list')
    return inner

def elgendi(sampling_rate: int = 1000) -> Callable[[list], dict]:
    def inner(signal): 
        df, _ = nk.ecg_process(signal, sampling_rate=sampling_rate, method='elgendi2010')
        return df.to_dict('list')
    return inner

def engzeemod(sampling_rate: int = 1000) -> Callable[[list], dict]:
    def inner(signal): 
        df, _ = nk.ecg_process(signal, sampling_rate=sampling_rate, method='engzeemod2012')
        return df.to_dict('list')
    return inner

def neurokit(sampling_rate: int = 1000) -> Callable[[list], dict]:
    def inner(signal): 
        df, _ = nk.ecg_process(signal, sampling_rate=sampling_rate, method='neurokit')
        return df.to_dict('list')
    return inner
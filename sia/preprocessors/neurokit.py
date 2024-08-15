import warnings
import neurokit2 as nk

from typing import Callable

def pantompkins(sampling_rate: int = 1000) -> Callable[[list], dict]:
    """Compute ECG features using the Pan-Tompkins algorithm.

    Parameters
    ----------
    sampling_rate : int, optional
        The sampling rate of the ECG signal, by default 1000.
    
    Returns
    -------
    function
        A function that computes the ECG features using the Pan-Tompkins algorithm.
    """
    def inner(signal: list[float]): 
        warnings.filterwarnings("ignore")
        df, _ = nk.ecg_process(signal, sampling_rate=sampling_rate, method='pantompkins1985')
        warnings.filterwarnings("default")
        return df.to_dict('list')
    return inner

def hamilton(sampling_rate: int = 1000) -> Callable[[list], dict]:
    """Compute ECG features using the Hamilton algorithm.

    Parameters
    ----------
    sampling_rate : int, optional
        The sampling rate of the ECG signal, by default 1000.

    Returns
    -------
    function
        A function that computes the ECG features using the Hamilton algorithm.
    """
    def inner(signal: list[float]): 
        warnings.filterwarnings("ignore")
        df, _ = nk.ecg_process(signal, sampling_rate=sampling_rate, method='hamilton2002')
        warnings.filterwarnings("default")
        return df.to_dict('list')
    return inner

def elgendi(sampling_rate: int = 1000) -> Callable[[list], dict]:
    """Compute ECG features using the Elgendi algorithm.

    Parameters
    ----------
    sampling_rate : int, optional
        The sampling rate of the ECG signal, by default 1000.

    Returns
    -------
    function
        A function that computes the ECG features using the Elgendi algorithm.
    """
    def inner(signal: list[float]): 
        warnings.filterwarnings("ignore")
        df, _ = nk.ecg_process(signal, sampling_rate=sampling_rate, method='elgendi2010')
        warnings.filterwarnings("default")
        return df.to_dict('list')
    return inner

def engzeemod(sampling_rate: int = 1000) -> Callable[[list], dict]:
    """Compute ECG features using the Engzee Modified algorithm.

    Parameters
    ----------
    sampling_rate : int, optional
        The sampling rate of the ECG signal, by default 1000.

    Returns
    -------
    function
        A function that computes the ECG features using the Engzee Modified algorithm.
    """
    def inner(signal: list[float]): 
        warnings.filterwarnings("ignore")
        df, _ = nk.ecg_process(signal, sampling_rate=sampling_rate, method='engzeemod2012')
        warnings.filterwarnings("default")
        return df.to_dict('list')
    return inner

def neurokit(sampling_rate: int = 1000) -> Callable[[list], dict]:
    """Compute ECG features using the NeuroKit algorithm.

    Parameters
    ----------
    sampling_rate : int, optional
        The sampling rate of the ECG signal, by default 1000.
    
    Returns
    -------
    function
        A function that computes the ECG features using the NeuroKit algorithm.
    """
    def inner(signal: list[float]): 
        warnings.filterwarnings("ignore")
        df, _ = nk.ecg_process(signal, sampling_rate=sampling_rate, method='neurokit')
        warnings.filterwarnings("default")
        return df.to_dict('list')
    return inner
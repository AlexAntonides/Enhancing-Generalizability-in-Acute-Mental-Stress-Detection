import warnings
from warnings import warn

import pandas as pd 

from neurokit2.misc import NeuroKitWarning
from neurokit2.signal import signal_psd
from neurokit2.signal.signal_power import _signal_power_continuous
from neurokit2.hrv.hrv_frequency import _hrv_format_input
from neurokit2.hrv.intervals_process import intervals_process

try:
    import cupy as cp
    np = cp
except ImportError:
    import numpy as np
    
from enum import Enum

class Feature(str, Enum):
    MEAN = 'mean'
    """The mean of the feature."""
    STD = 'std'
    """The standard deviation of the feature."""
    MIN = 'min'
    """The minimum of the feature."""
    MAX = 'max'
    """The maximum of the feature."""
    POWER = 'power'
    """The power of the feature."""
    COVARIANCE = 'covariance'
    """The covariance of the feature."""
    ENERGY = 'energy'
    """The energy of the feature."""
    ENTROPY = 'entropy'
    """The entropy of the feature."""

def frequency_domain(
        features: tuple[Feature], 
        ulf: tuple[int, int] = (0, 0.0033),
        vlf: tuple[int, int] = (0.0033, 0.04),
        lf: tuple[int, int] = (0.04, 0.15),
        hf: tuple[int, int] = (0.15, 0.4),
        vhf: tuple[int, int] = (0.4, 0.5),
        uhf: tuple[int, int] = (0.5, 1),
        sampling_rate: int = 1000
    ):
    """Compute frequency domain features from R-peaks.

    Parameters
    ----------
    features : tuple[Feature]
        A tuple of features to compute. Can be any of 'mean', 'std', 'min', 'max', 'power', 'covariance', 'energy', 'entropy'.
    ulf : tuple, optional
        Upper and lower limit of the ultra-low frequency band. By default (0, 0.0033).
    vlf : tuple, optional
        Upper and lower limit of the very-low frequency band. By default (0.0033, 0.04).
    lf : tuple, optional
        Upper and lower limit of the low frequency band. By default (0.04, 0.15).
    hf : tuple, optional
        Upper and lower limit of the high frequency band. By default (0.15, 0.4).
    vhf : tuple, optional
        Upper and lower limit of the very-high frequency band. By default (0.4, 0.5).
    uhf : tuple, optional   
        Upper and lower limit of the ultra-high frequency band. By default (0.5, 1).
    sampling_rate : int, optional
        Sampling rate (Hz) of the continuous cardiac signal in which the peaks occur. By default 1000.
    
    Returns
    -------
    function
        A function that takes in rpeaks and returns a dictionary of frequency domain features.
    """
    def inner(rpeaks: list[int]):
        result = {}
        warnings.filterwarnings("ignore")
        for feature in features:
            df = hrv_frequency(rpeaks, sampling_rate=sampling_rate, statistic=feature, ulf=ulf, vlf=vlf, lf=lf, hf=hf, vhf=vhf, uhf=uhf)
            df = df.fillna(0.00001)
            result.update({
                f'ulf_{feature}': df['HRV_ULF'].item(),
                f'vlf_{feature}': df['HRV_VLF'].item(),
                f'lf_{feature}': df['HRV_LF'].item(),
                f'hf_{feature}': df['HRV_HF'].item(),
                f'vhf_{feature}': df['HRV_VHF'].item(),
                f'uhf_{feature}': df['HRV_UHF'].item(),
                f'tp_{feature}': df['HRV_TP'].item(),
                f'lp_ulf_{feature}': (df['HRV_ULF'].item() / df['HRV_TP'].item()) * 100,
                f'lp_vlf_{feature}': (df['HRV_VLF'].item() / df['HRV_TP'].item()) * 100,
                f'lp_lf_{feature}': (df['HRV_LF'].item() / df['HRV_TP'].item()) * 100,
                f'lp_hf_{feature}': (df['HRV_HF'].item() / df['HRV_TP'].item()) * 100,
                f'lp_vhf_{feature}': (df['HRV_VHF'].item() / df['HRV_TP'].item()) * 100,
                f'lp_uhf_{feature}': (df['HRV_UHF'].item() / df['HRV_TP'].item()) * 100,
                f'lf_hf_{feature}': df['HRV_LF'].item() / df['HRV_HF'].item(),
            })
        warnings.filterwarnings("default")
        return result
    return inner

## 
# Neurokit Modification
## 
def hrv_frequency(
    peaks,
    sampling_rate=1000,
    ulf=(0, 0.0033),
    vlf=(0.0033, 0.04),
    lf=(0.04, 0.15),
    hf=(0.15, 0.4),
    vhf=(0.4, 0.5),
    uhf=(0.5, 1),
    psd_method="welch",
    silent=True,
    normalize=True,
    order_criteria=None,
    interpolation_rate=100,
    statistic: Feature ="power",
    **kwargs
):
    """**Computes frequency-domain indices of Heart Rate Variability (HRV)**

    Computes frequency domain HRV metrics, such as the power in different frequency bands.

    * **ULF**: The spectral power of ultra low frequencies (by default, .0 to
      .0033 Hz). Very long signals are required for this to index to be
      extracted, otherwise, will return NaN.
    * **VLF**: The spectral power of very low frequencies (by default, .0033 to .04 Hz).
    * **LF**: The spectral power of low frequencies (by default, .04 to .15 Hz).
    * **HF**: The spectral power of high frequencies (by default, .15 to .4 Hz).
    * **VHF**: The spectral power of very high frequencies (by default, .4 to .5 Hz).
    * **TP**: The total spectral power.
    * **LFHF**: The ratio obtained by dividing the low frequency power by the high frequency power.
    * **LFn**: The normalized low frequency, obtained by dividing the low frequency power by
      the total power.
    * **HFn**: The normalized high frequency, obtained by dividing the low frequency power by
      the total power.
    * **LnHF**: The log transformed HF.

    Note that a minimum duration of the signal containing the peaks is recommended for some HRV
    indices to be meaningful. For instance, 1, 2 and 5 minutes of high quality signal are the
    recommended minima for HF, LF and LF/HF, respectively.

    .. tip::

      We strongly recommend checking our open-access paper `Pham et al. (2021)
      <https://doi.org/10.3390/s21123998>`_ on HRV indices for more information.


    Parameters
    ----------
    peaks : dict
        Samples at which cardiac extrema (i.e., R-peaks, systolic peaks) occur.
        Can be a list of indices or the output(s) of other functions such as :func:`.ecg_peaks`,
        :func:`.ppg_peaks`, :func:`.ecg_process` or :func:`.bio_process`.
        Can also be a dict containing the keys `RRI` and `RRI_Time`
        to directly pass the R-R intervals and their timestamps, respectively.
    sampling_rate : int, optional
        Sampling rate (Hz) of the continuous cardiac signal in which the peaks occur.
    ulf : tuple, optional
        Upper and lower limit of the ultra-low frequency band. By default (0, 0.0033).
    vlf : tuple, optional
        Upper and lower limit of the very-low frequency band. By default (0.0033, 0.04).
    lf : tuple, optional
        Upper and lower limit of the low frequency band. By default (0.04, 0.15).
    hf : tuple, optional
        Upper and lower limit of the high frequency band. By default (0.15, 0.4).
    vhf : tuple, optional
        Upper and lower limit of the very-high frequency band. By default (0.4, 0.5).
    psd_method : str
        Method used for spectral density estimation. For details see :func:`.signal_power`.
        By default ``"welch"``.
    silent : bool
        If ``False``, warnings will be printed. Default to ``True``.
    show : bool
        If ``True``, will plot the power in the different frequency bands.
    normalize : bool
        Normalization of power by maximum PSD value. Default to ``True``.
        Normalization allows comparison between different PSD methods.
    order_criteria : str
        The criteria to automatically select order in parametric PSD (only used for autoregressive
        (AR) methods such as ``"burg"``). Defaults to ``None``.
    interpolation_rate : int, optional
        Sampling rate (Hz) of the interpolated interbeat intervals. Should be at least twice as
        high as the highest frequency in vhf. By default 100. To replicate Kubios defaults, set to 4.
        To not interpolate, set interpolation_rate to None (in case the interbeat intervals are already
        interpolated or when using the ``"lombscargle"`` psd_method for which interpolation is not required).
    statistic : Feature
        The statistic to compute. Can be one of 'min', 'max', 'mean', 'median', 'std', 'power', 'covariance', 'energy', 'entropy'.
    **kwargs
        Additional other arguments.

    Returns
    -------
    DataFrame
        Contains frequency domain HRV metrics.

    See Also
    --------
    ecg_peaks, ppg_peaks, hrv_summary, hrv_time, hrv_nonlinear

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      # Download data
      data = nk.data("bio_resting_5min_100hz")

      # Find peaks
      peaks, info = nk.ecg_peaks(data["ECG"], sampling_rate=100)

      # Compute HRV indices using method="welch"
      @savefig p_hrv_freq1.png scale=100%
      hrv_welch = nk.hrv_frequency(peaks, sampling_rate=100, show=True, psd_method="welch")
      @suppress
      plt.close()

    .. ipython:: python

      # Using method ="burg"
      @savefig p_hrv_freq2.png scale=100%
      hrv_burg = nk.hrv_frequency(peaks, sampling_rate=100, show=True, psd_method="burg")
      @suppress
      plt.close()

    .. ipython:: python

      # Using method = "lomb" (requires installation of astropy)
      @savefig p_hrv_freq3.png scale=100%
      hrv_lomb = nk.hrv_frequency(peaks, sampling_rate=100, show=True, psd_method="lomb")
      @suppress
      plt.close()

    .. ipython:: python

      # Using method="multitapers"
      @savefig p_hrv_freq4.png scale=100%
      hrv_multitapers = nk.hrv_frequency(peaks, sampling_rate=100, show=True,psd_method="multitapers")
      @suppress
      plt.close()

    References
    ----------
    * Pham, T., Lau, Z. J., Chen, S. H. A., & Makowski, D. (2021). Heart Rate Variability in
      Psychology: A Review of HRV Indices and an Analysis Tutorial. Sensors, 21(12), 3998.
    * Stein, P. K. (2002). Assessing heart rate variability from real-world Holter reports. Cardiac
      electrophysiology review, 6(3), 239-244.
    * Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and
      norms. Frontiers in public health, 5, 258.
    * Boardman, A., Schlindwein, F. S., & Rocha, A. P. (2002). A study on the optimum order of
      autoregressive models for heart rate variability. Physiological measurement, 23(2), 325.
    * Bachler, M. (2017). Spectral Analysis of Unevenly Spaced Data: Models and Application in Heart
      Rate Variability. Simul. Notes Eur., 27(4), 183-190.

    """

    # Sanitize input
    # If given peaks, compute R-R intervals (also referred to as NN) in milliseconds
    rri, rri_time, _ = _hrv_format_input(peaks, sampling_rate=sampling_rate)

    # Process R-R intervals (interpolated at 100 Hz by default)
    rri, rri_time, sampling_rate = intervals_process(
        rri, intervals_time=rri_time, interpolate=True, interpolation_rate=interpolation_rate, **kwargs
    )

    if interpolation_rate is None:
        t = rri_time
    else:
        t = None

    frequency_band = [ulf, vlf, lf, hf, vhf, uhf]

    # Find maximum frequency
    max_frequency = np.max([np.max(i) for i in frequency_band])

    power = signal_power(
        rri,
        frequency_band=frequency_band,
        sampling_rate=sampling_rate,
        method=psd_method,
        max_frequency=max_frequency,
        normalize=normalize,
        order_criteria=order_criteria,
        t=t,
        statistic=statistic
    )

    power.columns = ["ULF", "VLF", "LF", "HF", "VHF", "UHF"]

    out = power.to_dict(orient="index")[0]

    if silent is False:
        for frequency in out.keys():
            if out[frequency] == 0.0:
                warn(
                    "The duration of recording is too short to allow"
                    " reliable computation of signal power in frequency band " + frequency + "."
                    " Its power is returned as zero.",
                    category=NeuroKitWarning,
                )

    # Normalized
    total_power = np.nansum(power.values)
    out["TP"] = total_power
    out["LFHF"] = out["LF"] / out["HF"]
    out["LFn"] = out["LF"] / total_power
    out["HFn"] = out["HF"] / total_power
    out["LF/HF+LF"] = out["LF"] / (out["HF"] + out["LF"])
    out["HF/HF+LF"] = out["HF"] / (out["HF"] + out["LF"])

    # Log
    out["LnHF"] = np.log(out["HF"])  # pylint: disable=E1111

    out = pd.DataFrame.from_dict(out, orient="index").T.add_prefix(f"HRV_")
    return out

def signal_power(
    signal,
    frequency_band,
    sampling_rate=1000,
    continuous=False,
    normalize=True,
    statistic: Feature ="power",
    **kwargs,
):
    """**Compute the power of a signal in a given frequency band**

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    frequency_band :tuple or list
        Tuple or list of tuples indicating the range of frequencies to compute the power in.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    continuous : bool
        Compute instant frequency, or continuous power.
    show : bool
        If ``True``, will return a Poincaré plot. Defaults to ``False``.
    normalize : bool
        Normalization of power by maximum PSD value. Default to ``True``.
        Normalization allows comparison between different PSD methods.
    statistic : Feature
        The statistic to compute. Can be one of 'min', 'max', 'mean', 'median', 'std', 'power', 'covariance', 'energy', 'entropy'.
    **kwargs
        Keyword arguments to be passed to :func:`.signal_psd`.

    See Also
    --------
    signal_filter, signal_psd

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the Power Spectrum values and a plot if
        ``show`` is ``True``.

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk
      import numpy as np

      # Instant power
      signal = nk.signal_simulate(duration=60, frequency=[10, 15, 20],
                                  amplitude = [1, 2, 3], noise = 2)

      @savefig p_signal_power1.png scale=100%
      power_plot = nk.signal_power(signal, frequency_band=[(8, 12), (18, 22)], method="welch", show=True)
      @suppress
      plt.close()

    ..ipython:: python

      # Continuous (simulated signal)
      signal = np.concatenate((nk.ecg_simulate(duration=30, heart_rate=75), nk.ecg_simulate(duration=30, heart_rate=85)))
      power = nk.signal_power(signal, frequency_band=[(72/60, 78/60), (82/60, 88/60)], continuous=True)
      processed, _ = nk.ecg_process(signal)
      power["ECG_Rate"] = processed["ECG_Rate"]

      @savefig p_signal_power2.png scale=100%
      nk.signal_plot(power, standardize=True)
      @suppress
      plt.close()

    .. ipython:: python

      # Continuous (real signal)
      signal = nk.data("bio_eventrelated_100hz")["ECG"]
      power = nk.signal_power(signal, sampling_rate=100, frequency_band=[(0.12, 0.15), (0.15, 0.4)], continuous=True)
      processed, _ = nk.ecg_process(signal, sampling_rate=100)
      power["ECG_Rate"] = processed["ECG_Rate"]

      @savefig p_signal_power3.png scale=100%
      nk.signal_plot(power, standardize=True)
      @suppress
      plt.close()

    """

    if continuous is False:
        out = _signal_power_instant(
            signal,
            frequency_band,
            sampling_rate=sampling_rate,
            normalize=normalize,
            statistic=statistic,
            **kwargs,
        )
    else:
        out = _signal_power_continuous(signal, frequency_band, sampling_rate=sampling_rate)

    out = pd.DataFrame.from_dict(out, orient="index").T

    return out

def _signal_power_instant(
    signal,
    frequency_band,
    sampling_rate=1000,
    normalize=True,
    order_criteria="KIC",
    statistic="power",
    **kwargs,
):
    # Sanitize frequency band
    if isinstance(frequency_band[0], (int, float)):
        frequency_band = [frequency_band]  # put in list to iterate on

    #  Get min-max frequency
    min_freq = min([band[0] for band in frequency_band])
    max_freq = max([band[1] for band in frequency_band])

    # Get PSD
    psd = signal_psd(
        signal,
        sampling_rate=sampling_rate,
        show=False,
        normalize=normalize,
        order_criteria=order_criteria,
        **kwargs,
    )

    psd = psd[(psd["Frequency"] >= min_freq) & (psd["Frequency"] <= max_freq)]

    out = {}
    for band in frequency_band:
        if statistic == "min":
            value = _signal_min_instant_compute(psd, band)
        elif statistic == "max":
            value = _signal_max_instant_compute(psd, band)
        elif statistic == "mean":
            value = _signal_mean_instant_compute(psd, band)
        elif statistic == "median":
            value = _signal_median_instant_compute(psd, band)
        elif statistic == "std":
            value = _signal_std_instant_compute(psd, band)
        elif statistic == "power":
            value = _signal_power_instant_compute(psd, band)
        elif statistic == "covariance":
            value = _signal_covariance_instant_compute(psd, band)
        elif statistic == "energy": 
            value = _signal_energy_instant_compute(psd, band)
        elif statistic == "entropy": 
            value = _signal_entropy_instant_compute(psd, band)
        else:
            print("NeuroKit error: signal_power(): 'statistic' should be one of 'min', 'max', 'mean', 'median', 'std', 'power', 'covariance', 'energy', 'entropy'")
        out[f"Hz_{band[0]}_{band[1]}"] = value
    return out

def _signal_min_instant_compute(psd, band):
    """Calculates the minimum power in a given frequency band."""
    where = (psd["Frequency"] >= band[0]) & (psd["Frequency"] < band[1])
    min = np.min(psd["Power"][where])
    return np.nan if min == 0.0 else min

def _signal_max_instant_compute(psd, band):
    """Calculates the maximum power in a given frequency band."""
    where = (psd["Frequency"] >= band[0]) & (psd["Frequency"] < band[1])
    max = np.max(psd["Power"][where])
    return np.nan if max == 0.0 else max

def _signal_mean_instant_compute(psd, band):
    """Calculates the mean power in a given frequency band."""
    where = (psd["Frequency"] >= band[0]) & (psd["Frequency"] < band[1])
    mean = np.mean(psd["Power"][where])
    return np.nan if mean == 0.0 else mean

def _signal_median_instant_compute(psd, band):
    """Calculates the median power in a given frequency band."""
    where = (psd["Frequency"] >= band[0]) & (psd["Frequency"] < band[1])
    median = np.median(psd["Power"][where])
    return np.nan if median == 0.0 else median

def _signal_std_instant_compute(psd, band):
    """Calculates the standard deviation of power in a given frequency band."""
    where = (psd["Frequency"] >= band[0]) & (psd["Frequency"] < band[1])
    std = np.std(psd["Power"][where])
    return np.nan if std == 0.0 else std

def _signal_power_instant_compute(psd, band):
    """Calculates the power in a given frequency band."""
    where = (psd["Frequency"] >= band[0]) & (psd["Frequency"] < band[1])
    power = np.trapz(y=psd["Power"][where], x=psd["Frequency"][where])
    return np.nan if power == 0.0 else power

def _signal_covariance_instant_compute(psd, band):
    """Calculates the covariance of power in a given frequency band."""
    where = (psd["Frequency"] >= band[0]) & (psd["Frequency"] < band[1])
    covariance = np.cov(psd["Power"][where])
    return np.nan if covariance == 0.0 else covariance

def _signal_energy_instant_compute(psd, band):
    """Calculates the energy of power in a given frequency"""
    where = (psd["Frequency"] >= band[0]) & (psd["Frequency"] < band[1])
    energy = np.sum(psd["Power"][where])
    return np.nan if energy == 0.0 else energy

def _signal_entropy_instant_compute(psd, band):
    """Calculates the entropy of power in a given frequency"""
    where = (psd["Frequency"] >= band[0]) & (psd["Frequency"] < band[1])
    entropy = -np.sum(psd["Power"][where] * np.log2(psd["Power"][where]))
    return np.nan if entropy == 0.0 else entropy

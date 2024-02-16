import warnings
warnings.filterwarnings("ignore")

import os
import datetime 
import multiprocessing

import mne
import neurokit2 as nk
import pandas as pd
import numpy as np

import contextlib
import joblib
from joblib import Parallel, delayed

import argparse
from tqdm import tqdm

METADATA_FILENAME = 'Timestamps_Merged.txt'
SAMPLING_FREQUENCY = 1000

num_cores = 4

def get_edf_file_paths(target):
    if os.path.isdir(target):
        file_paths = [os.path.join(target, f) for f in os.listdir(target) if f.endswith('.edf')]
    elif os.path.isfile(target):
        file_paths = [target]
    else:
        raise Exception('Target is not a file or directory.')

    return file_paths

def pre_process(df, participant, output, preprocessor, window):
    if preprocessor == 'pantompkins':
        processed_df, info = nk.ecg_process(df['signal'], sampling_rate=SAMPLING_FREQUENCY, method='pantompkins1985')
    elif preprocessor == 'hamilton':
        processed_df, info = nk.ecg_process(df['signal'], sampling_rate=SAMPLING_FREQUENCY, method='hamilton2002')
    elif preprocessor == 'elgendi':
        processed_df, info = nk.ecg_process(df['signal'], sampling_rate=SAMPLING_FREQUENCY, method='elgendi2010')
    elif preprocessor == 'engzeemod':
        processed_df, info = nk.ecg_process(df['signal'], sampling_rate=SAMPLING_FREQUENCY, method='engzeemod2012')
    else:
        processed_df, info = nk.ecg_process(df['signal'], sampling_rate=SAMPLING_FREQUENCY, method='neurokit')

    if window is not None: 
        for window_df, window_processed in zip(np.array_split(df, window), np.array_split(processed_df, window)):
            # hrv_indices = nk.hrv(window_processed['ECG_R_Peaks'], sampling_rate=SAMPLING_FREQUENCY)
            hrv_time = nk.hrv_time(window_processed['ECG_R_Peaks'], sampling_rate=SAMPLING_FREQUENCY)
            hrv_freq = nk.hrv_frequency(window_processed['ECG_R_Peaks'], sampling_rate=SAMPLING_FREQUENCY, normalize=True)
            # hrv_nonlinear = nk.hrv_nonlinear(window_processed['ECG_R_Peaks'], sampling_rate=SAMPLING_FREQUENCY)

            # merged = pd.concat([window_df, window_processed, hrv_time, hrv_freq, hrv_nonlinear, hrv_indices], axis=1)
            merged = pd.concat([window_df, window_processed, hrv_time, hrv_freq], axis=1)

            if not os.path.exists(f'{output}/{participant}.csv'):
                merged.to_csv(f'{output}/{participant}.csv', index=False)
            else: 
                merged.to_csv(f'{output}/{participant}.csv', index=False, mode='a', header=False)
    else:
        merged = pd.concat([df, processed_df], axis=1)

        if not os.path.exists(f'{output}/{participant}.csv'):
            merged.to_csv(f'{output}/{participant}.csv', index=False)
        else: 
            merged.to_csv(f'{output}/{participant}.csv', index=False, mode='a', header=False)

def process_file(path, output, metadata, preprocessor, window):
    participant = os.path.splitext(os.path.basename(path))[0][:5]

    df = mne.io.read_raw_edf(path, preload=True, verbose=False)
    signals = df.get_data()[0]

    start_time = df.annotations.orig_time
    timestamps = [start_time + datetime.timedelta(seconds=delta) for delta in df.times]
    meta = metadata.loc[metadata['subject_id']==participant,:].reset_index(drop=True)

    meta['label_start'] = pd.to_datetime(meta['label_start'], utc=True)
    meta['label_end'] = pd.to_datetime(meta['label_end'], utc=True)
    
    df = pd.DataFrame({'timestamp': timestamps, 'signal': signals})
    df['signal_normalised'] = (df['signal'] - df['signal'].min()) / (df['signal'].max() - df['signal'].min())

    # Assuming your existing DataFrame has the columns (category, label_start, label_end)
    # Merge the two DataFrames on the timestamp column
    result_df = pd.merge_asof(df.sort_values('timestamp'), meta, left_on='timestamp', right_on='label_start', direction='backward')

    # Drop unnecessary columns
    result_df = result_df.drop(columns=['label_start', 'label_end'])

    result_df['category'].fillna('', inplace=True)
    result_df['subject_id'].fillna(participant, inplace=True)

    if preprocessor is not None:
        pre_process(result_df, participant, output, preprocessor, window)
    else:
        if not os.path.exists(f'{output}/{participant}.csv'):
            result_df.to_csv(f'{output}/{participant}.csv', index=False)
        else: 
            result_df.to_csv(f'{output}/{participant}.csv', index=False, mode='a', header=False)

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='To_CSV',
        description='Processes .EDF files, pre-processes the data, and turns them into .CSV files.',
        epilog='Written by Alex Antonides.'
    )

    parser.add_argument('directory', type=str, help='the directory to process')
    parser.add_argument('output', type=str, help='the output target')

    parser.add_argument('--metadata', '-m', type=str, help='the metadata file to use', required=False)
    parser.add_argument('--preprocessor', '-p', choices=['neurokit', 'pantompkins', 'hamilton', 'elgendi', 'engzeemod'], help='preprocesses the file with the given method when set')
    parser.add_argument('--window', '-w', type=int)

    args = parser.parse_args()

    directory = args.directory
    output = args.output

    print(f'Processing directory {directory}.')

    file_paths = get_edf_file_paths(directory)

    if args.metadata:
        metadata_path = args.metadata
    else:
        metadata_path = os.path.join(directory, METADATA_FILENAME)

    metadata = pd.read_csv(metadata_path, sep='\t', decimal=',', skiprows=[0], header=None, names=['subject_id', 'category', 'code', 'label_start', 'label_end'], dtype={'subject_id': 'str', 'category': 'str', 'code': 'str', 'label_start': 'str', 'label_end': 'str'})

    if not os.path.exists(output):
        os.mkdir(output)
        
    with tqdm_joblib(tqdm(total=len(file_paths))) as progress_bar:
        Parallel(n_jobs=num_cores)(delayed(process_file)(path, output, metadata, args.preprocessor, args.window) for path in file_paths)
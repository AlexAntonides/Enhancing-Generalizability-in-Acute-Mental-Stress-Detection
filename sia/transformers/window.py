from typing import Callable

def as_window(window_size: int = 1000) -> Callable[[list], dict]:
    def inner(signal, idxs, dataset):
        return {
            'window': [dataset[idx:idx+window_size]['signal'] if len(dataset[idx:idx+window_size]['signal']) == window_size else [] for idx in idxs]
        }
    return inner
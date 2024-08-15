"""Segmenters module for Stress-in-Action."""
import sys

from abc import ABC
from collections.abc import Iterable

from datasets import Dataset

from typing import Any
if sys.version_info >= (3,11):
    from typing import Self 
else:
    Self = Any

class BaseSegmenter(ABC, Iterable):
    """Base class for segmenters."""
    def set_dataset(self, dataset: Dataset) -> Self:
        """Set the dataset to segment.
        
        Parameters
        ----------
        dataset : Dataset
            The dataset to segment.
        
        Returns
        -------
        Self
            The instance of the class for method
        """
        raise NotImplementedError

class SlidingWindow(BaseSegmenter):
    """A sliding window segmenter."""
    def __init__(self, window_size: int, step_size: int = None):
        """
        Parameters
        ----------
        window_size : int
            The size of the window.
        step_size : int, optional
            The step size, by default, it is the same as the window size.
        """
        self.dataset = None
        self.window_size = window_size
        self.step_size = step_size or window_size

    def set_dataset(self, dataset: Dataset) -> Self:
        self.dataset = dataset
        return self

    def __len__(self):
        if self.dataset is None:
            return 0
        else:
            return len(self.dataset) // self.step_size

    def __iter__(self):
        for start_idx in range(0, len(self.dataset), self.step_size):
            # If the window is too short, skip it
            if start_idx + self.window_size > len(self.dataset):
                break
            yield self.dataset[start_idx:start_idx + self.window_size]
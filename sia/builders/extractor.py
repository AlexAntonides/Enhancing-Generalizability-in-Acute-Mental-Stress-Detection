import sys
from typing import Any, TypeVar, Iterator, Union, Callable, Generator

T = TypeVar('T')
if sys.version_info >= (3,11):
    from typing import Self 
else:
    Self = Any

import datasets
from datasets import Dataset

from tqdm.auto import tqdm

from .preprocessing import Pipeline, Preprocessing
from ..segmenters import BaseSegmenter

class Segmenter(Preprocessing):
    """
    The segmenter pipeline class in the SiA-Kit acts as a middleman between the data and the feature extractor, and is responsible for dividing the data into smaller parts.
    """
    __SEGMENT_SETTING_KEY = 'segment'

    def segment(self, method: BaseSegmenter) -> Self:
        """Segmentation is the process of dividing the data into smaller parts. It is primarily done by a segmentation method, which dictates how the data is divided based on given parameters.

        Parameters
        ----------
        method : BaseSegmenter
            The segmentation method, which dictates how the data is divided based on given parameters.

        Returns
        -------
        Self
            The feature extractor.  
        """
        segmenter = _Extractor(self, method, num_proc=self._num_proc)
        self[self.__SEGMENT_SETTING_KEY] = segmenter
        return segmenter

    ## --- Pipeline Executors --- ##
    def _execute(self, key: str, value: Union[Any, list]) -> None:
        """Executes the given key-value pair in the pipeline settings.
        
        Parameters
        ----------
        key : str
            The key of the setting to execute
        value : Union[Any, list]
            The value of the setting to execute
        """
        try:
            # First try the custom executors.
            self.__execute(key, value)
        except KeyError:
            # If the key is not found, try the parent executors.
            super()._execute(key, value)
            
    def __execute(self, key: str, value: Union[Any, list]) -> None:
        """Executes the given key-value pair in the pipeline settings.
        
        Parameters
        ----------
        key : str
            The key of the setting to execute
        value : Union[Any, list]
            The value of the setting to execute
        """
        if key == self.__SEGMENT_SETTING_KEY:
            with tqdm(total=len(self._data_store), desc='Segmenting...', unit='files', position=1) as pbar:
                datasets.disable_progress_bars()
                for i, data in enumerate(self._data_store):
                    segments = []
                    for segment in value._segment(data['dataset']):
                        segments.append(segment)
                    self._data_store[i]['dataset'] = Dataset.from_list(segments)
                    pbar.update(1)
                datasets.enable_progress_bars()
        else:
            raise KeyError(f"Invalid key: {key}")

class SkipIteration(Exception): pass

class _Extractor(Segmenter):
    """
    The internal extractor pipeline class in the SiA-Kit is responsible for extracting features from the data.
    The extractor cannot be used directly, but must be retrieved from the segmenter, considering features are only extracted from segments.
    """
    __EXTRACT_SETTING_KEY = 'extract'
    __SKIP_SETTING_KEY = 'skip'
    __USE_SETTING_KEY = 'use'

    def __init__(self, previous: Pipeline, segmenter: Callable[..., Generator], num_proc: int = 8):
        """Initializes the internal extractor pipeline class in the SiA-Kit.
        
        Parameters
        ----------
        previous : Pipeline
            The previous pipeline in the pipeline chain. Considering the extractor is only used to extract features from segments, the previous pipeline is 
            to eventually redirect the pipeline back into the main pipeline chain.
        segmenter : Callable[..., Generator]
            The segmenter method, which dictates how the data is divided based on given parameters.
        num_proc : int, optional
            The number of processes to use, by default 8 
        """
        super().__init__(num_proc=num_proc)
        self._previous_pipeline = previous
        self.__segmenter = segmenter

    def extract(self, a: Union[str, Callable], b: Callable = None) -> Self:
        """Extracts features from the data. The extraction is done by a given method, which dictates how the features are extracted based on given parameters.
        
        Parameters
        ----------
        a : Union[str, Callable]
            Either the name of the new column to store the extracted features in, or the method to extract the features.
        b : Callable, optional
            When the first parameter is the name of the new column, this parameter is assumed to be the method to extract the features, thus, by default None

        Returns
        -------
        Self
            The feature extractor.
        """
        data = {}
        if b == None:
            data['method'] = a
        else:
            data['column'] = a
            data['method'] = b

        self[self.__EXTRACT_SETTING_KEY] = data
        return self

    def skip(self, callable: Callable):
        """Skips the current iteration if the given condition is met.

        Parameters
        ----------
        callable : Callable
            The condition to skip the current iteration.
        
        Returns
        -------
        Self
            The feature extractor.  
        """
        self[self.__SKIP_SETTING_KEY] = callable
        return self

    def use(self, name: str, callable: Callable) -> Self:
        """Stores the result of the given method in memory, to be used in other methods.

        Parameters
        ----------
        name : str
            The name of the memory to store the result in.
        callable : Callable
            The method to store the result of in memory.

        Returns
        -------
        Self
            The feature extractor.
        """
        self[self.__USE_SETTING_KEY] = {name: callable}
        return self
    
    def _segment(self, data: Dataset) -> Iterator[Dataset]:
        """Use the extractor to segment the data, and extract features from the segments.

        Parameters
        ----------
        data : Dataset
            The data to segment.

        Returns
        -------
        Iterator[Dataset]
            The segmented data.
        """
        def _get_args(fn) -> list:
            """Get the arguments of a given function
            
            Parameters
            ----------
            fn : Callable
                The function to get the arguments from

            Returns
            -------
            list
                The arguments of the function
            """
            return fn.__code__.co_varnames[:fn.__code__.co_argcount]
        
        segmenter = self.__segmenter.set_dataset(data)
        with tqdm(total=len(segmenter), desc='Extracting Features...', unit='windows', position=2) as pbar:
            for segment in segmenter:
                memory = {} # Memory for storing results from the use method.
                extracted_features = {}
                try:
                    for setting in self._settings:
                        key, value = next(iter(setting.items()))

                        if key == self.__EXTRACT_SETTING_KEY:
                            method = value['method']

                            args = _get_args(method)
                            features = self.__segmenter.dataset.features.keys()
                            memory_features = memory.keys()
                            intersection = list(set(features) & set(args) | set(memory_features) & set(args))

                            kwargs = {}
                            if "dataset" in args:
                                kwargs["dataset"] = self.__segmenter.dataset
                            
                            args = []
                            for column in intersection:
                                if column in memory_features:
                                    args.append(memory[column])
                                else:
                                    args.append(segment[column])
                            
                            if 'column' in value:
                                extracted_features[value['column']] = method(*args, **kwargs)
                            else:
                                extracted_features.update(method(*args, **kwargs))
                        elif key == self.__SKIP_SETTING_KEY:
                            args = _get_args(value)
                            features = self.__segmenter.dataset.features.keys()
                            memory_features = memory.keys()
                            intersection = list(set(features) & set(args) | set(memory_features) & set(args))

                            kwargs = {}
                            if "dataset" in args:
                                kwargs["dataset"] = self.__segmenter.dataset
                            
                            args = []
                            for column in intersection:
                                if column in memory_features:
                                    args.append(memory[column])
                                else:
                                    args.append(segment[column])

                            if value(*args, **kwargs):
                                raise SkipIteration()
                        elif key == self.__USE_SETTING_KEY:
                            for name, method in value.items():
                                args = _get_args(method)
                                features = self.__segmenter.dataset.features.keys()
                                memory_features = memory.keys()
                                intersection = list(set(features) & set(args) | set(memory_features) & set(args))
                                kwargs = {}
                                if "dataset" in args:
                                    kwargs["dataset"] = self.__segmenter.dataset
                                
                                args = []
                                for column in intersection:
                                    if column in memory_features:
                                        args.append(memory[column])
                                    else:
                                        args.append(segment[column])

                                memory[name] = method(*args, **kwargs)
                    pbar.update(1)
                    yield extracted_features
                except SkipIteration:
                    pbar.update(1)
                    continue
            pbar.total = pbar.n
            pbar.refresh()

    def __len__(self) -> int:
        return len(self.__segmenter)

    def __getattribute__(self, name: str):
        """This method is used to redirect the method calls to the previous pipeline in the pipeline chain, when a method is used that is not defined in the extractor."""
        if name.startswith('__') or name.startswith('__Extractor__'):
            return object.__getattribute__(self, name)
        try:
            attr = object.__getattribute__(self, name)
            if callable(attr) and hasattr(Segmenter, name) and name not in _Extractor.__dict__:
                def wrapper(*args, **kwargs):
                    previous_instance = object.__getattribute__(self, '_previous_pipeline')
                    return object.__getattribute__(previous_instance, name)(*args, **kwargs)
                return wrapper
            return attr
        except AttributeError:
            previous_instance = object.__getattribute__(self, '_previous_pipeline')
            return object.__getattribute__(previous_instance, name)
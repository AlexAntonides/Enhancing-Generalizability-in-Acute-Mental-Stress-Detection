import sys
from pathlib import Path
from prettytable import PrettyTable

from typing import Any, TypeVar, Iterator, Union, Callable, Tuple

T = TypeVar('T')
if sys.version_info >= (3,11):
    from typing import Self 
else:
    Self = Any

import datasets
from datasets import Dataset, IterableDataset

from sklearn.preprocessing._encoders import _BaseEncoder

from tqdm.auto import tqdm

from .pipeline import Pipeline

class Preprocessing(Pipeline):
    """
    The preprocessing pipeline class in the SiA-Kit to preprocess datasets.
    """
    __READER_SETTING_KEY = 'reader'
    __PROCESS_SETTING_KEY = 'process'
    __FILTER_SETTING_KEY = 'filter'
    __SELECT_SETTING_KEY = 'select'
    __DROP_SETTING_KEY = 'drop'
    __RENAME_SETTING_KEY = 'rename'
    __ENCODER_SETTING_KEY = 'encoder'

    def __init__(self, num_proc: int = 8):
        """Initialize the Preprocessing pipeline.

        Parameters
        ----------
            num_proc (int): The number of processes to use for multiprocessing.
        """
        super().__init__()
        self._num_proc = num_proc
        self._data_store = []

    ## --- Pipeline Configurators --- ##
    def data(self, reader: Callable[..., Callable[[], Iterator[Tuple[str, Union[Dataset, IterableDataset]]]]]) -> Self:
        """Configuration to read data using a given reader.
            
        Parameters
        ----------
        reader : Callable[..., Callable[[], Iterator[Tuple[str, Union[Dataset, IterableDataset]]]]
            The reader function to read the data.
            A reader function is a high-order function that yields the path of the file, and a Huggingface Dataset containing the data of the file.
            Keep in mind that the data is batched, and the reader function should be able to handle batched data.

        Returns
        -------
        Self
            The instance of the class for method chaining

        Examples
        --------

        ```python
        def reader(path: str):
            def inner():
                for path in Path(path).rglob('*.csv'):
                    yield path, Dataset.from_pandas(pd.read_csv(path))
            return inner

        Preprocessing() \\ 
        .data(reader('./data'))
        ```
        """
        self[self.__READER_SETTING_KEY] = reader
        return self

    def process(self, processor: Callable[..., Callable[..., dict]]) -> Self:
        """Configuration to process data using a given processor.
        
        Parameters
        ----------
        processor : Callable[..., Callable[..., dict]] 
            The processor function to process the data.
            A processor function is a high-order function that takes any columns of the dataset, and returns a dictionary new or modified columns.
            Keep in mind that the data is batched, and the processor function should be able to handle batched data.

        Returns
        -------
        Self
            The instance of the class for method chaining

        Examples
        --------
        
        ```python
        def processor(sampling_rate: int):
            def inner(signal: list):
                clean = clean_signal(signal, sampling_rate=sampling_rate)
                return {'raw': signals, 'clean': clean}
            return inner

        Preprocessing() \\
        .process(processor(1000))
        ```
        """
        self[self.__PROCESS_SETTING_KEY] = processor
        return self
    
    def filter(self, filter: Callable[..., Tuple[bool]]) -> Self:
        """Configuration to filter data using a given filter.
        
        Parameters
        ----------
        filter : Callable[..., Tuple[bool]]) 
            The filter function to filter the data.
            A filter function is a function that takes any columns of the dataset as input, and returns a boolean value to filter the data.
            Keep in mind that the data is batched, and the filter function should be able to handle batched data.

        Returns
        -------
        Self
            The instance of the class for method chaining

        Examples
        --------
        >>> Preprocessing().filter(lambda signal: [x > 0.5 for x in signal])
        """
        self[self.__FILTER_SETTING_KEY] = filter
        return self
    
    def select(self, columns: Union[str, list[str]]) -> Self:
        """Configuration to select columns from the dataset.
        
        Parameters
        ----------
        columns : Union[str, list[str]] 
            The columns to select from the dataset.

        Returns
        -------
        Self
            The instance of the class for method chaining

        Examples
        --------
        >>> Preprocessing().select(['timestamp', 'signal'])
        """
        setting = []
        if isinstance(columns, str):
            setting.append(columns)
        else: 
            setting.extend(columns)
        self[self.__SELECT_SETTING_KEY] = setting
        return self
    
    def drop(self, columns: Union[str, list[str]]) -> Self:
        """Configuration to drop columns from the dataset.
        
        Parameters
        ----------
        columns : Union[str, list[str]] 
            The columns to drop from the dataset.

        Returns
        -------
        Self
            The instance of the class for method chaining

        Examples
        --------
        >>> Preprocessing().drop(['timestamp'])
        """
        setting = []
        if isinstance(columns, str):
            setting.append(columns)
        else: 
            setting.extend(columns)
        self[self.__DROP_SETTING_KEY] = setting
        return self
    
    def rename(self, x: Union[str, dict, list[str, dict]], y: str = None) -> Self:
        """Configuration to rename columns in the dataset.
        
        Parameters
        ----------
        x : Union[str, dict, list[str, dict]])
            The target column to rename, or a combination of the target column and the new name
        y : str, optional 
            The new name of the column, if x is a string

        Returns
        -------
        Self
            The instance of the class for method chaining

        Examples
        --------
        >>> Preprocessing().rename('signal', 'ecg')

        >>> Preprocessing().rename({'signal': 'ecg'})
        """
        setting = []
        def _rename(x: Union[str, dict, list[str, dict]], y: str = None) -> list[dict]:
            if y is not None:
                if isinstance(x, str):
                    return [{x: y}]
                else: 
                    raise ValueError('Invalid input')
            else:
                if isinstance(x, dict):
                    columns = []
                    for key, value in x.items():
                        columns.append({ f"{key}": value })
                    return columns
                elif isinstance(x, list):
                    if len(x) == 2 and isinstance(x[0], str):
                        return _rename(x[0], x[1])
                    elif isinstance(x[0], dict):
                        return _rename(x)
                    else: 
                        raise ValueError('Invalid input')
                else: 
                    raise ValueError('Invalid input')

        setting.extend(_rename(x, y))
        self[self.__RENAME_SETTING_KEY] = setting
        return self
    
    def encode(self, column: Union[str, dict[str, str]], encoder: _BaseEncoder) -> Self:
        """Configuration to encode columns in the dataset.

        Parameters
        ----------
        column : Union[str, dict[str, str]]
            The target column to encode, or a combination of the target column and the new name
        encoder : _BaseEncoder
            The encoder to encode the column

        Returns
        -------
        Self
            The instance of the class for method chaining

        Examples
        --------
        >>> Preprocessing().encode('label', sklearn.preprocessing.OneHotEncoder())

        >>> Preprocessing().encode({'label': 'category'}, sia.encoders.GroupEncoder())
        """
        self[self.__ENCODER_SETTING_KEY] = { 'column': column, 'encoder': encoder }
        return self
    
    ## --- Pipeline Executors --- ##
    def to(self, writer: Callable[..., Callable[[str, Dataset], None]]) -> None:
        """Execute the pipeline and write the dataset using a given writer.
        
        Parameters
        ----------
        writer : Callable[..., Callable[[str, Dataset], None]
            The writer function to write the data.
            A writer function is a high-order function that takes the filename and the dataset, and writes the data to a file.

        Examples
        --------
        
        ```python
        def to_csv(path: str):
            def inner(filename: str, dataset: Dataset):
                dataset.to_csv(path)
            return inner

        Preprocessing() \\
        .data(reader('./data/file.csv')) \\
        .write(to_csv('./output.csv'))
        ```
        """
        with tqdm(total=len(self) + 1, desc='Pipeline', unit='steps', position=0, leave=True) as pbar1:
            for key, value in self:
                self._execute(key, value)
                pbar1.update(1)
            
            with tqdm(total=len(self._data_store), desc='Writing...', unit='files', position=1) as pbar2:
                datasets.disable_progress_bars()
                for data in self._data_store:
                    filename = Path(data["path"]).stem
                    writer(filename, data["dataset"])
                pbar2.update(1)
                datasets.enable_progress_bars()
            pbar1.update(1)

    def _execute(self, key: str, value: Union[Any, list]) -> None:
        """Executes the given key-value pair in the pipeline settings.
        
        Parameters
        ----------
        key : str
            The key of the setting to execute
        value : Union[Any, list]
            The value of the setting to execute
        """
        self.__execute(key, value)

    def __execute(self, key: str, value: Union[Any, list]) -> None:
        """Executes the given key-value pair in the pipeline settings.
        
        Parameters
        ----------
        key : str
            The key of the setting to execute
        value : Union[Any, list]
            The value of the setting to execute
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
        
        if isinstance(value, list):
            for _ in value:
                self._execute(key, _)
        else:
            if key == self.__READER_SETTING_KEY:
                self._data_store = []
                with tqdm(total=1, desc='Reading...', unit='files', position=1, bar_format='{desc} |{bar}| {n_fmt} files read [{elapsed}, {rate_fmt}{postfix}]') as pbar:
                    datasets.disable_progress_bars()
                    for path, dataset in value():
                        self._data_store.append({ "path": path, "dataset": dataset })
                        pbar.total = pbar.total + 1
                        pbar.update(1)
                    pbar.total = pbar.total - 1
                    pbar.refresh()
                    datasets.enable_progress_bars()
            elif key == self.__PROCESS_SETTING_KEY:
                args = _get_args(value)
                with tqdm(total=len(self._data_store), desc='Processing...', unit='files', position=1) as pbar:
                    datasets.disable_progress_bars()
                    for data in self._data_store:
                        features = data["dataset"].features.keys()
                        intersection = list(set(features) & set(args))

                        kwargs = {}
                        if "dataset" in args:
                            kwargs["dataset"] = data["dataset"]

                        data["dataset"] = data["dataset"].map(
                            value,
                            batched=True,
                            batch_size=int(len(data["dataset"]) * .25),
                            input_columns=intersection,
                            with_indices='idxs' in args,
                            fn_kwargs=kwargs,
                            num_proc=self._num_proc
                        )
                        pbar.update(1)
                    datasets.enable_progress_bars()
            elif key == self.__SELECT_SETTING_KEY:
                with tqdm(total=len(self._data_store), desc='Selecting Columns...', unit='files', position=1) as pbar:
                    datasets.disable_progress_bars()
                    for data in self._data_store:
                        data["dataset"] = data["dataset"].select_columns(value)
                        pbar.update(1)
                    datasets.enable_progress_bars()
            elif key == self.__DROP_SETTING_KEY:
                with tqdm(total=len(self._data_store), desc='Dropping Columns...', unit='files', position=1) as pbar:
                    datasets.disable_progress_bars()
                    for data in self._data_store:
                        data["dataset"] = data["dataset"].remove_columns(value)
                        pbar.update(1)
                    datasets.enable_progress_bars()
            elif key == self.__RENAME_SETTING_KEY:
                with tqdm(total=len(self._data_store), desc='Renaming Columns...', unit='files', position=1) as pbar:
                    datasets.disable_progress_bars()
                    for data in self._data_store:
                        for _from, _to in value.items():
                            data["dataset"] = data["dataset"].rename_columns({ f'{_from}': _to })
                        pbar.update(1)
                    datasets.enable_progress_bars()
            elif key == self.__FILTER_SETTING_KEY:
                args = _get_args(value)
                with tqdm(total=len(self._data_store), desc='Filtering...', unit='files', position=1) as pbar:
                    datasets.disable_progress_bars()
                    for data in self._data_store:
                        features = data["dataset"].features.keys()
                        intersection = list(set(features) & set(args))

                        kwargs = {}
                        if "dataset" in args:
                            kwargs["dataset"] = data["dataset"]

                        data["dataset"] = data["dataset"].filter(
                            value,
                            batched=True,
                            batch_size=int(len(data["dataset"]) * .25),
                            input_columns=intersection,
                            with_indices='idxs' in args,
                            fn_kwargs=kwargs,
                            num_proc=self._num_proc
                        )
                        pbar.update(1)
                    datasets.enable_progress_bars()
            elif key == self.__ENCODER_SETTING_KEY:
                with tqdm(total=len(self._data_store), desc='Encoding...', unit='files', position=1) as pbar:
                    datasets.disable_progress_bars()
                    for data in self._data_store:
                        _column = value["column"]
                        _encoder = value["encoder"] 

                        if type(_column) == str:
                            _from = _to = _column
                        else:
                            _from, _to = next(iter(_column.items()))

                        data["dataset"] = data["dataset"].map(
                            lambda x: { _to: _encoder.fit_transform(x[_from]) },
                            batched=True,
                            batch_size=int(len(data["dataset"]) * .25),
                            num_proc=self._num_proc
                        )
                        pbar.update(1)
                    datasets.enable_progress_bars()
            else:
                raise KeyError(f"Invalid key: {key}")
            
    def __repr__(self, table=None) -> str:
        """Generate a representation of the Preprocessing pipeline settings.
        
        Parameters
        ----------
        table : PrettyTable, optional
            The table to add the representation to, by default PrettyTable()

        Returns
        -------
        str
            A formatted table representing all settings.
        """
        if table is None:
            table = PrettyTable()
            
        table.add_row(['num_proc', self._num_proc])
        table = super().__repr__(table)
        return str(table)

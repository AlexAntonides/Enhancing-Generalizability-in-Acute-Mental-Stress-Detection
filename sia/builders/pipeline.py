import sys

import collections
from abc import ABC
from typing import Any, List, TypeVar

T = TypeVar('T')
if sys.version_info >= (3,11):
    from typing import Self 
else:
    Self = Any

from prettytable import PrettyTable

class Pipeline(ABC, collections.abc.Iterator):
    """
    The base class for all pipelines in the SiA-Kit.

    A pipeline is a series of human-readable commands that are executed in order that they are given.
    It consists of a list of configurators and executors, where configurators are used to set the pipeline settings and executors are used to execute the pipeline
    
    This abstract class allows a pipeline to inherit the basic functionalities, i.e. setting and getting pipeline commands.
    """

    def __init__(self):
        self._settings = []

    def __setitem__(self, key: str, value: T) -> Self:
        """Set a key-value pair in the pipeline settings.

        Parameters
        ----------
        key : str
            The key for the setting.
        value : T 
            The value for the setting.

        Returns
        -------
        Self
            The instance of the class for method chaining.
        """
        self._settings.append({key: value})
        return self
    
    def __delitem__(self, key: str) -> Self:
        """Delete a key-value pair from the pipeline settings.

        Parameters
        ----------
        key : str
            The key to delete.

        Returns
        -------
        Self
            The instance of the class for method chaining.
        """
        self._settings = [{k: v for k, v in _ if k != key} for _ in self._settings]
        return self
    
    def __getitem__(self, key: str) -> List[T]:
        """Get all values associated with a given key.

        Parameters
        ----------
        key : str
            The key to search for.

        Returns
        -------
        List[T]
            A list of all values associated with the given key.
        """
        return [_[key] for _ in self._settings if key in _]

    def __next__(self) -> tuple[str, any]:
        """Get the next setting in the pipeline settings.
        
        Returns
        -------
        key, value
            The key-value pair of the next setting.
        """
        try:
            setting = self._settings.pop(0)
            return list(setting)[0], setting[list(setting)[0]]
        except IndexError:
            raise StopIteration  
        
    def __repr__(self, table=None) -> PrettyTable:
        """Generate a representation of the Pipeline settings.

        Returns
        -------
        PrettyTable
            A formatted table representing all settings.
        """
        if table == None:
            table = PrettyTable()

        if table.title == None:
            table.title = f'{type(self).__name__} Settings'

        table.field_names = ['Setting', 'Value(s)']

        settings = self._settings.copy()
        while True:
            try:
                table.add_row(self.__next__())
            except StopIteration:
                break
        self._settings = settings

        return table
    
    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the pipeline settings.

        Parameters
        ----------
        key : str 
            The key to search for.

        Returns
        -------
        bool
            True if the key exists, False otherwise.
        """
        return any(key in _ for _ in self._settings)

    def __len__(self) -> int:
        """Get the number of settings in the pipeline.

        Returns
        -------
        int
            The number of settings in the pipeline.
        """
        return len(self._settings)
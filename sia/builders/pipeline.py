import sys

from abc import ABC
from typing import Any, List, TypeVar

from prettytable import PrettyTable

T = TypeVar('T')
if sys.version_info >= (3,11):
    from typing import Self 
else:
    Self = Any

class Pipeline(ABC):
    """
    The base class for all pipelines in the SiA-Kit.

    A pipeline is a series of human-readable commands that are executed in order that they are given.
    It consists of a list of configurators and executors, where configurators are used to set the pipeline settings and executors are used to execute the pipeline
    
    This abstract class allows a pipeline to inherit the basic functionalities, i.e. setting and getting pipeline commands.
    """

    def __init__(self):
        self.__settings = []

    def __setitem__(self, key: str, value: T) -> Self:
        """
        Set a key-value pair in the pipeline settings.

        Args:
            key (str): The key for the setting.
            value (T): The value for the setting.

        Returns:
            Self: The instance of the class for method chaining.
        """
        self.__settings.append({key: value})
        return self
    
    def __delitem__(self, key: str) -> Self:
        """
        Delete a key-value pair from the pipeline settings.

        Args:
            key (str): The key to delete.

        Returns:
            Self: The instance of the class for method chaining.
        """
        self.__settings = [{k: v for k, v in _ if k != key} for _ in self.__settings]
        return self
    
    def __getitem__(self, key: str) -> List[T]:
        """
        Get all values associated with a given key.

        Args:
            key (str): The key to search for.

        Returns:
            List[T]: A list of all values associated with the given key.
        """
        return [_[key] for _ in self.__settings if key in _]

    def __iter__(self):
        """
        Initialize the iterator for the pipeline settings.
        """
        return self

    def __next__(self):
        """
        Get the next setting in the pipeline settings.
        
        Returns:
            key, value: The key-value pair of the next setting.
        """
        try:
            setting = self.__settings.pop(0)
            return list(setting)[0]
        except IndexError:
            raise StopIteration  
        
    def __repr__(self, table=PrettyTable()):
        """
        Generate a representation of the Pipeline settings.

        Returns:
            PrettyTable: A formatted table representing all settings.
        """
        if table.title == None:
            table.title = f'{type(self).__name__} Settings'

        table.field_names = ['Setting', 'Value(s)']
        for _, setting in self._iterate():
            key_in_table = False
            for key, value in setting.items():
                if key_in_table == False:
                    table.add_row([key, value])
                else: 
                    table.add_row(['', value])

        return table
    
    def __contains__(self, key: str):
        """
        Check if a key exists in the pipeline settings.

        Args:
            key (str): The key to search for.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        return any(key in _ for _ in self.__settings)

    def __len__(self):
        """
        Get the number of settings in the pipeline.

        Returns:
            int: The number of settings in the pipeline.
        """
        return len(self.__settings)
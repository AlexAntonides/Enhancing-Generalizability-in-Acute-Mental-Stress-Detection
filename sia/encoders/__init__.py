"""Encoders module for Stress-in-Action."""

from typing import Iterator, Union

from sklearn.base import OneToOneFeatureMixin
from sklearn.preprocessing._encoders import _BaseEncoder

class GroupEncoder(OneToOneFeatureMixin, _BaseEncoder):
    """
    While standard encoders are designed to encode a string feature into a numerical one, this encoder is designed to "encode" a string into a group of categories.
    """

    def __init__(self, groups: dict[str, list]):
        """
        Parameters
        ----------
        groups : dict
            A dictionary where the keys are the groups and the values are lists of strings that belong to that groups.
        """
        self.groups = groups

    def fit(self, X: Union[list, Iterator[str]], y: None = None):
        """
        Fit the encoder to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to determine the groups of each feature.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        Returns
        -------
        self
            Fitted encoder.
        """
        # No fitting necessary, considering the groups are given.
        return self

    def transform(self, X: Union[list, Iterator[str]]):
        """
        Transform the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to encode.

        Returns
        -------
        X_out : {ndarray, sparse matrix} of shape \
                (n_samples, n_encoded_features)
            Transformed input. 
        """
        def encode(x: str) -> list:
            for key, list in self.groups.items():
                if x in list:
                    return key
            return None
        return [encode(x) for x in X]
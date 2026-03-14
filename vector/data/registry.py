"""Dataset registry with decorator-based loader registration."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class DatasetInfo:
    """Metadata about a dataset."""

    name: str
    n_dims: int
    window_size: int
    n_sequences: int


@dataclass
class SequenceData:
    """Single sequence from a dataset.

    All arrays are 2D: train/test have shape (T, D) where D >= 1.
    Univariate data is stored as (T, 1) -- never as 1D arrays.
    """

    name: str
    train: np.ndarray  # shape (T_train, D)
    test: np.ndarray  # shape (T_test, D)
    labels: np.ndarray  # shape (T_test,) or (T_test, D)


class BaseLoader:
    """Base class for dataset loaders.

    Subclasses implement load() to return a list of SequenceData.
    Register via the @register decorator.
    """

    def load(self, data_dir: str) -> List[SequenceData]:
        """Load raw data and return list of sequences."""
        raise NotImplementedError


DATASET_REGISTRY: Dict[str, BaseLoader] = {}


def register(name: str):
    """Decorator that registers a loader class in DATASET_REGISTRY.

    Usage:
        @register("NAB")
        class NABLoader(BaseLoader):
            def load(self, data_dir): ...
    """

    def decorator(cls):
        DATASET_REGISTRY[name] = cls()
        return cls

    return decorator

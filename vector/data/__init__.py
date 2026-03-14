"""Data loading, preprocessing, and configuration for VECTOR."""

from vector.data.registry import (
    DATASET_REGISTRY,
    BaseLoader,
    DatasetInfo,
    SequenceData,
    register,
)
from vector.data.preprocess import (
    normalize_sequence,
    normalize_splits,
    temporal_split,
    create_windows,
)
from vector.data.config import (
    load_config,
    get_dataset_config,
    DEFAULT_CONFIG_PATH,
)

__all__ = [
    "DATASET_REGISTRY",
    "BaseLoader",
    "DatasetInfo",
    "SequenceData",
    "register",
    "normalize_sequence",
    "normalize_splits",
    "temporal_split",
    "create_windows",
    "load_config",
    "get_dataset_config",
    "DEFAULT_CONFIG_PATH",
]

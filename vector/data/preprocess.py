"""Shared preprocessing pipeline: normalization, splitting, windowing."""

from typing import Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler


def normalize_sequence(
    train: np.ndarray, test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Z-score normalize using train-only statistics.

    Fits StandardScaler on train, transforms both train and test.
    No information leaks from test into the scaler.

    Args:
        train: Training data, shape (T_train, D).
        test: Test data, shape (T_test, D).

    Returns:
        Normalized train, normalized test, fitted scaler.
    """
    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train)
    test_norm = scaler.transform(test)
    return train_norm, test_norm, scaler


def normalize_splits(
    train: np.ndarray, val: np.ndarray, test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Z-score normalize train/val/test using train-only statistics.

    Args:
        train: Training data, shape (T_train, D).
        val: Validation data, shape (T_val, D).
        test: Test data, shape (T_test, D).

    Returns:
        Normalized train, val, test, and fitted scaler.
    """
    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train)
    val_norm = scaler.transform(val)
    test_norm = scaler.transform(test)
    return train_norm, val_norm, test_norm, scaler


def temporal_split(
    train: np.ndarray, val_ratio: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """Split training data temporally -- last val_ratio becomes validation.

    Preserves temporal ordering. No shuffling.

    Args:
        train: Training data, shape (T, D).
        val_ratio: Fraction of train to use as validation (default 0.2).

    Returns:
        (train_split, val_split) where train_split is the first (1-val_ratio)
        and val_split is the last val_ratio of the input.
    """
    split_idx = int(len(train) * (1 - val_ratio))
    return train[:split_idx], train[split_idx:]


def create_windows(
    data: np.ndarray, window_size: int, stride: int = 1
) -> np.ndarray:
    """Create sliding windows from time-series data.

    Args:
        data: Input array, shape (T, D).
        window_size: Number of time steps per window.
        stride: Step size between consecutive windows (default 1).

    Returns:
        Windows array, shape (N, window_size, D) where
        N = (T - window_size) // stride + 1.
        If T < window_size, returns empty array with shape (0, window_size, D).
    """
    n_timesteps, n_dims = data.shape

    if n_timesteps < window_size:
        return np.empty((0, window_size, n_dims), dtype=data.dtype)

    n_windows = (n_timesteps - window_size) // stride + 1
    windows = np.empty((n_windows, window_size, n_dims), dtype=data.dtype)

    for i in range(n_windows):
        start = i * stride
        windows[i] = data[start : start + window_size]

    return windows

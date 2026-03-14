"""Shared fixtures for data pipeline tests."""

import os

import numpy as np
import pytest

from vector.data.registry import SequenceData


@pytest.fixture
def dummy_data_dir(tmp_path):
    """Create a temp directory with small dummy SWaT-like data for fast tests.

    Returns path to directory containing train.npy, test.npy, labels.npy
    with shapes (200, 5), (100, 5), (100,).
    """
    n_train, n_test, n_dims = 200, 100, 5
    rng = np.random.RandomState(0)

    train = rng.randn(n_train, n_dims)
    test = rng.randn(n_test, n_dims)
    labels = np.zeros(n_test, dtype=np.int32)
    labels[30:50] = 1  # anomaly segment

    np.save(os.path.join(tmp_path, "train.npy"), train)
    np.save(os.path.join(tmp_path, "test.npy"), test)
    np.save(os.path.join(tmp_path, "labels.npy"), labels)

    return str(tmp_path)


@pytest.fixture
def sample_sequence():
    """Return a SequenceData with known shapes for unit tests."""
    rng = np.random.RandomState(1)
    return SequenceData(
        name="test_seq",
        train=rng.randn(200, 5),
        test=rng.randn(100, 5),
        labels=np.zeros(100, dtype=np.int32),
    )


@pytest.fixture
def config_override():
    """Return a minimal datasets.yaml-style config dict for testing."""
    return {
        "datasets": {
            "TestDS": {
                "raw_path": "/tmp/fake_raw",
                "processed_path": "/tmp/fake_processed",
                "window_size": 10,
                "n_dims": 5,
                "n_sequences": 1,
                "format": "npy",
            }
        }
    }

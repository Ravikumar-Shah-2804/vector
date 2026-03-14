"""SMD (Server Machine Dataset) loader with interpretation_label parsing."""

import os
import warnings
from typing import List

import numpy as np

from vector.data.registry import BaseLoader, SequenceData, register


def _parse_interpretation_labels(label_path: str, n_timesteps: int, n_dims: int) -> np.ndarray:
    """Parse SMD interpretation_label file into per-dimension binary labels.

    Each line has format: "start-end:col1,col2,col3" with 1-based indices.

    Args:
        label_path: Path to the interpretation_label file.
        n_timesteps: Number of timesteps in the test sequence.
        n_dims: Number of dimensions in the test sequence.

    Returns:
        1D binary labels of shape (n_timesteps,), reduced from per-dim via max.
    """
    labels_2d = np.zeros((n_timesteps, n_dims), dtype=np.float64)

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            range_part, cols_part = line.split(":")
            start_str, end_str = range_part.split("-")
            start = int(start_str) - 1  # convert to 0-based
            end = int(end_str)  # end is exclusive after -1 + 1

            col_indices = [int(c) - 1 for c in cols_part.split(",")]
            labels_2d[start:end, col_indices] = 1.0

    return labels_2d.max(axis=1)


@register("SMD")
class SMDLoader(BaseLoader):
    """Loader for Server Machine Dataset (28 sequences, 38 dims)."""

    def load(self, data_dir: str) -> List[SequenceData]:
        train_dir = os.path.join(data_dir, "train")
        test_dir = os.path.join(data_dir, "test")
        label_dir = os.path.join(data_dir, "interpretation_label")

        filenames = sorted(f for f in os.listdir(train_dir) if f.endswith(".txt"))

        sequences: List[SequenceData] = []
        for filename in filenames:
            name = filename[: -len(".txt")]

            train = np.genfromtxt(
                os.path.join(train_dir, filename), dtype=np.float64, delimiter=","
            )
            test = np.genfromtxt(
                os.path.join(test_dir, filename), dtype=np.float64, delimiter=","
            )

            label_path = os.path.join(label_dir, filename)
            if os.path.exists(label_path):
                labels = _parse_interpretation_labels(
                    label_path, test.shape[0], test.shape[1]
                )
            else:
                warnings.warn(
                    f"No interpretation_label file for {name}, using all-zero labels"
                )
                labels = np.zeros(test.shape[0], dtype=np.float64)

            sequences.append(
                SequenceData(name=name, train=train, test=test, labels=labels)
            )

        return sequences

"""MBA dataset loader for bivariate Excel-format heartbeat data."""

import os
from typing import List

import numpy as np
import pandas as pd

from vector.data.registry import BaseLoader, SequenceData, register


@register("MBA")
class MBALoader(BaseLoader):
    """Load MBA (MIT-BIH Arrhythmia) dataset from Excel files.

    Expects data_dir to contain:
        - train.xlsx: training data (skip first column = index)
        - test.xlsx: test data (skip first column = index)
        - labels.xlsx: anomaly point indices

    Produces a single SequenceData with 2D arrays of shape (T, 2).
    """

    def load(self, data_dir: str) -> List[SequenceData]:
        """Load MBA train/test/labels from Excel files."""
        train_path = os.path.join(data_dir, "train.xlsx")
        test_path = os.path.join(data_dir, "test.xlsx")
        labels_path = os.path.join(data_dir, "labels.xlsx")

        # Load train: skip first column (index), keep remaining as float64
        train_df = pd.read_excel(train_path, header=0, engine="openpyxl")
        train = train_df.iloc[:, 1:].values.astype(np.float64)

        # Load test: same format
        test_df = pd.read_excel(test_path, header=0, engine="openpyxl")
        test = test_df.iloc[:, 1:].values.astype(np.float64)

        # Load labels: contains indices of anomaly points
        labels_df = pd.read_excel(labels_path, header=0, engine="openpyxl")
        anomaly_indices = labels_df.iloc[:, 0].values.astype(int)

        # Create binary labels array for test data
        labels = np.zeros(len(test), dtype=np.float64)
        # Clip indices to valid range
        valid_indices = anomaly_indices[anomaly_indices < len(test)]
        valid_indices = valid_indices[valid_indices >= 0]
        labels[valid_indices] = 1.0

        return [
            SequenceData(
                name="MBA",
                train=train,
                test=test,
                labels=labels,
            )
        ]

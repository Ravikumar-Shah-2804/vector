"""UCR dataset loader with sequence selection and filename-encoded metadata."""

import os
import re
from typing import List

import numpy as np

from vector.data.registry import BaseLoader, SequenceData, register

SELECTED_IDS = [135, 136, 137, 138]


@register("UCR")
class UCRLoader(BaseLoader):
    """Load UCR anomaly detection dataset from raw TXT files.

    UCR filenames encode metadata:
        {id}_UCR_Anomaly_{name}_{train_end}_{anomaly_start}_{anomaly_end}.txt

    Only sequences in SELECTED_IDS are loaded (non-trivial InternalBleeding
    sequences from TransNAS_TSAD). Data is split at train_end index and
    labels are created from anomaly_start/anomaly_end ranges.
    """

    def load(self, data_dir: str) -> List[SequenceData]:
        """Load selected UCR sequences with filename-parsed splits and labels."""
        sequences: List[SequenceData] = []

        pattern = re.compile(r"^(\d+)_UCR_Anomaly_(.+)\.txt$")

        filenames = sorted(os.listdir(data_dir))
        for filename in filenames:
            match = pattern.match(filename)
            if match is None:
                continue

            file_id = int(match.group(1))
            if file_id not in SELECTED_IDS:
                continue

            # Parse metadata from filename: last 3 underscore-separated numbers
            # Format: {id}_UCR_Anomaly_{name}_{train_end}_{anomaly_start}_{anomaly_end}.txt
            stem = filename.replace(".txt", "")
            parts = stem.split("_")
            anomaly_end = int(parts[-1])
            anomaly_start = int(parts[-2])
            train_end = int(parts[-3])

            # Load full series
            filepath = os.path.join(data_dir, filename)
            full_data = np.loadtxt(filepath, dtype=np.float64)

            # Split at train_end (1-indexed in filename)
            train = full_data[:train_end].reshape(-1, 1)
            test = full_data[train_end:].reshape(-1, 1)

            # Create labels for test portion
            test_len = len(test)
            labels = np.zeros(test_len, dtype=np.float64)

            # Anomaly indices are 1-indexed in full series, convert to test-relative
            anomaly_start_test = anomaly_start - train_end
            anomaly_end_test = anomaly_end - train_end

            # Clip to valid test range
            start_idx = max(0, anomaly_start_test)
            end_idx = min(test_len, anomaly_end_test)
            if start_idx < end_idx:
                labels[start_idx:end_idx] = 1.0

            seq_name = match.group(2)
            # Include file ID in name for uniqueness
            name = f"{file_id}_{seq_name}"

            sequences.append(
                SequenceData(
                    name=name,
                    train=train,
                    test=test,
                    labels=labels,
                )
            )

        return sequences

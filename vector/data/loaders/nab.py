"""NAB dataset loader with nyc_taxi exclusion and timestamp-based labels."""

import json
import os
from typing import List

import numpy as np
import pandas as pd

from vector.data.registry import BaseLoader, SequenceData, register

EXCLUDED_TRACES = ["nyc_taxi"]


@register("NAB")
class NABLoader(BaseLoader):
    """Load NAB dataset from raw CSV files with JSON label mapping.

    NAB stores each trace as a CSV with columns (timestamp, value).
    Labels are in labels.json mapping trace paths to anomaly timestamps.
    Train and test are identical (full series) per NAB benchmark protocol.
    """

    def load(self, data_dir: str) -> List[SequenceData]:
        """Load NAB traces, excluding nyc_taxi, with timestamp-based labels."""
        labels_path = os.path.join(data_dir, "labels.json")
        with open(labels_path) as f:
            label_dict = json.load(f)

        sequences: List[SequenceData] = []

        # Walk subdirectories to find all CSV files
        csv_files: list[tuple[str, str]] = []
        for root, _dirs, files in os.walk(data_dir):
            for fname in sorted(files):
                if fname.endswith(".csv"):
                    csv_files.append((root, fname))

        csv_files.sort(key=lambda x: x[1])

        for dirpath, filename in csv_files:
            trace_name = filename.replace(".csv", "")

            # DATA-08: Skip excluded traces
            if any(excl in trace_name for excl in EXCLUDED_TRACES):
                continue

            filepath = os.path.join(dirpath, filename)
            df = pd.read_csv(filepath)
            values = df.iloc[:, 1].values.astype(np.float64)
            data = values.reshape(-1, 1)

            # Build binary labels from timestamp matching
            labels = np.zeros(len(values), dtype=np.float64)

            # Try to find matching key in label_dict
            # Keys use subdirectory prefix like "realKnownCause/filename.csv"
            matched_key = None
            for key in label_dict:
                if key.endswith(filename):
                    matched_key = key
                    break

            if matched_key is not None and label_dict[matched_key]:
                for timestamp in label_dict[matched_key]:
                    # Strip .000000 suffix before matching
                    tstamp = timestamp.replace(".000000", "")
                    matches = np.where(
                        df.iloc[:, 0].astype(str).str.replace(".000000", "", regex=False)
                        == tstamp
                    )[0]
                    if len(matches) > 0:
                        idx = matches[0]
                        start = max(0, idx - 4)
                        end = min(len(values), idx + 5)  # +5 because slice is exclusive
                        labels[start:end] = 1.0

            # NAB: train == test (full series, unsupervised evaluation)
            sequences.append(
                SequenceData(
                    name=trace_name,
                    train=data.copy(),
                    test=data.copy(),
                    labels=labels,
                )
            )

        return sequences

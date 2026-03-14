"""SMAP dataset loader with shared helper for SMAP/MSL datasets."""

import ast
import os
from typing import List

import numpy as np
import pandas as pd

from vector.data.registry import BaseLoader, SequenceData, register


def _load_smap_msl(data_dir: str, spacecraft: str) -> List[SequenceData]:
    """Load sequences for SMAP or MSL from their shared directory structure.

    Both datasets share labeled_anomalies.csv and the same train/test .npy layout.
    The spacecraft column distinguishes SMAP from MSL channels.

    Args:
        data_dir: Path to the SMAP_MSL raw data directory.
        spacecraft: Either 'SMAP' or 'MSL'.

    Returns:
        List of SequenceData, one per channel, sorted by chan_id.
    """
    labels_path = os.path.join(data_dir, "labeled_anomalies.csv")
    df = pd.read_csv(labels_path)
    df = df[df["spacecraft"] == spacecraft].sort_values("chan_id")

    sequences: List[SequenceData] = []
    for _, row in df.iterrows():
        chan_id = row["chan_id"]

        train = np.load(os.path.join(data_dir, "train", f"{chan_id}.npy"))
        test = np.load(os.path.join(data_dir, "test", f"{chan_id}.npy"))

        # Parse anomaly_sequences -- stored as string in CSV
        anomaly_ranges = ast.literal_eval(row["anomaly_sequences"])
        labels = np.zeros(test.shape[0], dtype=np.float64)
        for start, end in anomaly_ranges:
            labels[start:end] = 1.0

        sequences.append(
            SequenceData(name=chan_id, train=train, test=test, labels=labels)
        )

    return sequences


@register("SMAP")
class SMAPLoader(BaseLoader):
    """Loader for NASA SMAP soil moisture dataset (55 sequences, 25 dims)."""

    def load(self, data_dir: str) -> List[SequenceData]:
        return _load_smap_msl(data_dir, "SMAP")

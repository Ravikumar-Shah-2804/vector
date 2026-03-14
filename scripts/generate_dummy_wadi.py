"""Generate dummy WADI dataset for development without iTrust access.

Creates train.npy, test.npy, and labels.npy with 123 sensor dimensions.
Anomalies are injected as signal shifts in random dimension subsets.
"""

import os

import numpy as np


def generate_dummy_wadi(output_dir: str = "data/raw/WADI", seed: int = 42) -> None:
    """Generate dummy WADI data with injected anomalies.

    Args:
        output_dir: Directory to write .npy files to.
        seed: Random seed for reproducibility.
    """
    n_train = 5000
    n_test = 2000
    n_dims = 123

    rng = np.random.RandomState(seed)

    # Normal sensor readings
    train = rng.randn(n_train, n_dims)
    test = rng.randn(n_test, n_dims)
    labels = np.zeros(n_test, dtype=np.int32)

    # Inject 3 anomaly segments (more affected dims than SWaT)
    anomaly_starts = [200, 800, 1500]
    for start in anomaly_starts:
        length = rng.randint(50, 150)
        end = min(start + length, n_test)
        affected_dims = rng.choice(n_dims, size=8, replace=False)
        test[start:end, affected_dims] += 3.0
        labels[start:end] = 1

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "train.npy"), train)
    np.save(os.path.join(output_dir, "test.npy"), test)
    np.save(os.path.join(output_dir, "labels.npy"), labels)

    print(f"WADI dummy data saved to {output_dir}")
    print(f"  train: {train.shape}, test: {test.shape}, labels: {labels.shape}")
    print(f"  anomaly ratio: {labels.sum() / len(labels):.2%}")


if __name__ == "__main__":
    generate_dummy_wadi()

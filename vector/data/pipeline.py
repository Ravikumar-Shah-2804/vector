"""End-to-end preprocessing pipeline: load -> normalize -> split -> window -> save."""

import os
from typing import Dict, List, Optional

import numpy as np

from vector.data.config import get_dataset_config, load_config
from vector.data.preprocess import create_windows, normalize_sequence, temporal_split
from vector.data.registry import DATASET_REGISTRY


def preprocess_dataset(
    dataset_name: str,
    config: Optional[dict] = None,
    output_dir: Optional[str] = None,
) -> Optional[Dict]:
    """Preprocess a single dataset: load -> normalize -> split -> window -> save.

    Args:
        dataset_name: Registry key (e.g. "SWaT", "NAB").
        config: Pre-loaded datasets.yaml dict. Loaded from default path if None.
        output_dir: Override for output directory. Uses config processed_path if None.

    Returns:
        Summary dict with dataset name, sequence count, and shapes.
        None if the dataset could not be loaded (missing files).
    """
    if config is None:
        config = load_config()

    ds_cfg = get_dataset_config(dataset_name, config)

    if dataset_name not in DATASET_REGISTRY:
        # Ensure all loaders are registered
        import vector.data.loaders  # noqa: F401

    if dataset_name not in DATASET_REGISTRY:
        print(f"WARNING: Skipping {dataset_name} - no loader registered")
        return None

    loader = DATASET_REGISTRY[dataset_name]
    raw_dir = ds_cfg["raw_path"]

    try:
        sequences = loader.load(raw_dir)
    except (FileNotFoundError, OSError) as e:
        print(f"WARNING: Skipping {dataset_name} - {e}")
        return None

    save_dir = output_dir or ds_cfg["processed_path"]
    window_size = ds_cfg["window_size"]
    shapes: Dict[str, dict] = {}

    for seq in sequences:
        # Normalize: fit on train only
        train_norm, test_norm, _scaler = normalize_sequence(seq.train, seq.test)

        # Temporal split: last 20% of train becomes validation
        train_split, val_split = temporal_split(train_norm, val_ratio=0.2)

        # Create sliding windows
        train_windows = create_windows(train_split, window_size)
        val_windows = create_windows(val_split, window_size)
        test_windows = create_windows(test_norm, window_size)

        # Align labels with windowed test data: take label at last timestep of each window
        labels = seq.labels
        if labels.ndim > 1:
            # Reduce per-dimension labels to 1D via max
            labels = labels.max(axis=1)
        windowed_labels = labels[window_size - 1 :]
        # Truncate to match number of test windows
        windowed_labels = windowed_labels[: len(test_windows)]

        # Save to disk
        seq_dir = os.path.join(save_dir, seq.name)
        os.makedirs(seq_dir, exist_ok=True)

        np.save(os.path.join(seq_dir, "train.npy"), train_windows)
        np.save(os.path.join(seq_dir, "val.npy"), val_windows)
        np.save(os.path.join(seq_dir, "test.npy"), test_windows)
        np.save(os.path.join(seq_dir, "test_labels.npy"), windowed_labels)

        shapes[seq.name] = {
            "train": train_windows.shape,
            "val": val_windows.shape,
            "test": test_windows.shape,
            "labels": windowed_labels.shape,
        }

    return {
        "dataset": dataset_name,
        "n_sequences": len(sequences),
        "shapes": shapes,
    }


def preprocess_all(
    config: Optional[dict] = None,
    datasets: Optional[List[str]] = None,
) -> Dict[str, Optional[Dict]]:
    """Preprocess multiple datasets.

    Args:
        config: Pre-loaded datasets.yaml dict. Loaded from default path if None.
        datasets: List of dataset names to process. If None, processes all
            datasets defined in config.

    Returns:
        Dict mapping dataset name to its result (or None if skipped).
    """
    if config is None:
        config = load_config()

    if datasets is None:
        datasets = list(config.get("datasets", {}).keys())

    # Ensure all loaders are registered
    import vector.data.loaders  # noqa: F401

    results: Dict[str, Optional[Dict]] = {}

    for name in datasets:
        result = preprocess_dataset(name, config)
        results[name] = result
        if result is not None:
            seq_info = ", ".join(
                f"{sn}: train{s['train']}" for sn, s in result["shapes"].items()
            )
            print(f"  {name}: {result['n_sequences']} sequence(s) -- {seq_info}")
        else:
            print(f"  {name}: SKIPPED")

    processed = sum(1 for r in results.values() if r is not None)
    print(f"\nProcessed {processed}/{len(datasets)} datasets")

    return results

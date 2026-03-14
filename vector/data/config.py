"""Dataset configuration loading from datasets.yaml."""

import os
from typing import Dict, Optional

import yaml

DEFAULT_CONFIG_PATH = os.path.join("experiments", "configs", "datasets.yaml")


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load dataset configuration from YAML file.

    Args:
        config_path: Path to datasets.yaml. Defaults to
            experiments/configs/datasets.yaml.

    Returns:
        Parsed YAML as dict.

    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_dataset_config(
    dataset_name: str, config: Optional[dict] = None
) -> dict:
    """Get configuration for a single dataset.

    Args:
        dataset_name: Name of the dataset (e.g., "NAB", "SMD").
        config: Pre-loaded config dict. If None, loads from default path.

    Returns:
        Dict with dataset-specific configuration.

    Raises:
        KeyError: If dataset_name not found. Lists available datasets
            in the error message.
    """
    if config is None:
        config = load_config()

    datasets = config.get("datasets", {})

    if dataset_name not in datasets:
        available = ", ".join(sorted(datasets.keys()))
        raise KeyError(
            f"Dataset '{dataset_name}' not found. "
            f"Available datasets: {available}"
        )

    return datasets[dataset_name]

"""Search configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_search_config(path: str | Path | None = None) -> dict:
    """Load search configuration from YAML file.

    Parameters
    ----------
    path : str, Path, or None
        Path to the search config YAML file. Defaults to
        ``experiments/configs/search.yaml`` relative to the project root.

    Returns
    -------
    dict
        Parsed search configuration.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at the resolved path.
    """
    if path is None:
        project_root = Path(__file__).resolve().parents[2]
        path = project_root / "experiments" / "configs" / "search.yaml"
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Search config not found at {path}. "
            f"Expected experiments/configs/search.yaml in the project root."
        )

    with open(path, "r") as f:
        return yaml.safe_load(f)

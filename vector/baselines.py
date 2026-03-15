"""Baseline evaluation methods for VECTOR NAS comparison.

Three baselines reuse the identical objective() function from the search
engine (BASE-05 compliance): default fixed parameters, grid search over
rho x n_res, and random search via Optuna RandomSampler.
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path

import numpy as np
import optuna

from vector.search.objective import objective

logger = logging.getLogger(__name__)

# BASE-01: Default MD-RS hyperparameters
DEFAULT_PARAMS: dict = {
    "n_res": 500,
    "rho": 0.9,
    "sigma": 0.1,
    "sparsity": 0.1,
    "alpha": 0.3,
    "k": 1,
    "n_wash": 50,
}

# BASE-03: Grid search axes
GRID_RHOS = [0.3, 0.6, 0.9, 1.2]
GRID_N_RES = [100, 300, 500]


def run_default_baseline(
    sequences: list[dict],
    dataset_name: str,
    search_config: dict,
    dataset_config: dict,
) -> dict:
    """Evaluate the default MD-RS configuration (BASE-01).

    Creates a FixedTrial with DEFAULT_PARAMS and runs it through the
    standard objective function to guarantee identical evaluation.

    Returns
    -------
    dict
        method, f1, effective_size, params.
    """
    try:
        trial = optuna.trial.FixedTrial(DEFAULT_PARAMS)
        obj1, obj2 = objective(
            trial, sequences, dataset_name, search_config, dataset_config,
        )
        return {
            "method": "default",
            "f1": float(1.0 - obj1),
            "effective_size": float(obj2),
            "params": dict(DEFAULT_PARAMS),
        }
    except Exception:
        logger.exception("Default baseline failed for %s", dataset_name)
        return {
            "method": "default",
            "f1": 0.0,
            "effective_size": float(DEFAULT_PARAMS["n_res"]),
            "params": dict(DEFAULT_PARAMS),
        }


def run_grid_search_baseline(
    sequences: list[dict],
    dataset_name: str,
    search_config: dict,
    dataset_config: dict,
) -> dict:
    """Evaluate 12 grid configurations: 4 rho x 3 n_res (BASE-03).

    Remaining parameters are fixed to DEFAULT_PARAMS values.

    Returns
    -------
    dict
        method, best_f1, best_params, n_configs, all_configs.
    """
    fixed = {
        "sigma": DEFAULT_PARAMS["sigma"],
        "sparsity": DEFAULT_PARAMS["sparsity"],
        "alpha": DEFAULT_PARAMS["alpha"],
        "k": DEFAULT_PARAMS["k"],
        "n_wash": DEFAULT_PARAMS["n_wash"],
    }

    all_configs: list[dict] = []
    for rho, n_res in itertools.product(GRID_RHOS, GRID_N_RES):
        params = {**fixed, "rho": rho, "n_res": n_res}
        try:
            trial = optuna.trial.FixedTrial(params)
            obj1, obj2 = objective(
                trial, sequences, dataset_name, search_config, dataset_config,
            )
            all_configs.append({
                "params": params,
                "f1": float(1.0 - obj1),
                "effective_size": float(obj2),
            })
        except Exception:
            logger.debug(
                "Grid config rho=%.1f n_res=%d failed", rho, n_res,
                exc_info=True,
            )
            all_configs.append({
                "params": params,
                "f1": 0.0,
                "effective_size": float(n_res),
            })

    best = max(all_configs, key=lambda c: c["f1"])
    return {
        "method": "grid_search",
        "best_f1": best["f1"],
        "best_params": best["params"],
        "n_configs": len(all_configs),
        "all_configs": all_configs,
    }


def _to_serializable(obj: object) -> object:
    """Convert numpy types to native Python types for JSON safety."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _walk_serialize(data: object) -> object:
    """Recursively walk a structure and convert numpy types."""
    if isinstance(data, dict):
        return {k: _walk_serialize(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_walk_serialize(item) for item in data]
    return _to_serializable(data)


def save_baseline_results(
    results: dict,
    dataset_name: str,
    output_dir: str = "experiments/results",
) -> Path:
    """Save baseline results to JSON (BASE-04).

    Parameters
    ----------
    results : dict
        Combined results dict with keys for each baseline method.
    dataset_name : str
        Dataset name, used for subdirectory.
    output_dir : str
        Root output directory.

    Returns
    -------
    Path
        Path to the written JSON file.
    """
    combined = {"dataset": dataset_name, **results}
    serializable = _walk_serialize(combined)

    out_path = Path(output_dir) / dataset_name / "baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)

    logger.info("Baseline results saved to %s", out_path)
    return out_path

"""Baseline evaluation methods for VECTOR NAS comparison.

Three baselines reuse the identical objective() function from the search
engine (BASE-05 compliance): default fixed parameters, grid search over
rho x n_res, and random search via Optuna RandomSampler.
"""

from __future__ import annotations

import functools
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


def run_random_search_baseline(
    dataset_name: str,
    sequences: list[dict],
    search_config: dict,
    dataset_config: dict,
    n_trials: int = 1500,
    n_jobs: int = 4,
) -> dict:
    """Run random search baseline with Optuna RandomSampler (BASE-02).

    Uses in-memory storage (no SQLite) to avoid collision with the main
    VECTOR study database.

    Parameters
    ----------
    dataset_name : str
        Dataset name for study naming.
    sequences : list[dict]
        Sequence dicts with train/val/test/labels arrays.
    search_config : dict
        Parsed search.yaml configuration.
    dataset_config : dict
        Parsed datasets.yaml configuration.
    n_trials : int
        Number of random trials to run.
    n_jobs : int
        Parallel workers.

    Returns
    -------
    dict
        method, best_f1, n_trials, best_params.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(
        study_name=f"baseline_random_{dataset_name}",
        directions=["minimize", "minimize"],
        sampler=sampler,
    )

    objective_fn = functools.partial(
        objective,
        sequences=sequences,
        dataset_name=dataset_name,
        search_config=search_config,
        dataset_config=dataset_config,
    )

    study.optimize(objective_fn, n_trials=n_trials, n_jobs=n_jobs)

    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]

    if not completed:
        return {
            "method": "random_search",
            "best_f1": 0.0,
            "n_trials": len(study.trials),
            "best_params": {},
        }

    best_trial = min(completed, key=lambda t: t.values[0])
    best_f1 = 1.0 - best_trial.values[0]

    return {
        "method": "random_search",
        "best_f1": float(best_f1),
        "n_trials": len(study.trials),
        "best_params": dict(best_trial.params),
    }


def run_all_baselines(
    sequences: list[dict],
    dataset_name: str,
    search_config: dict,
    dataset_config: dict,
    output_dir: str = "experiments/results",
) -> dict:
    """Run all three baselines and save combined results.

    Convenience function that orchestrates default, grid search, and
    random search baselines, then serializes everything to JSON.

    Returns
    -------
    dict
        Combined results with keys: default, grid_search, random_search.
    """
    logger.info("Running baselines for %s", dataset_name)

    default_result = run_default_baseline(
        sequences, dataset_name, search_config, dataset_config,
    )
    grid_result = run_grid_search_baseline(
        sequences, dataset_name, search_config, dataset_config,
    )
    random_result = run_random_search_baseline(
        dataset_name, sequences, search_config, dataset_config,
    )

    results = {
        "default": default_result,
        "grid_search": grid_result,
        "random_search": random_result,
    }

    save_baseline_results(results, dataset_name, output_dir)

    return results

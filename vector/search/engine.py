"""Search engine orchestration: study factory and optimization runner."""

from __future__ import annotations

import functools
import os
from pathlib import Path

import optuna

from vector.search.objective import objective


def create_or_load_study(
    dataset_name: str,
    search_config: dict,
) -> optuna.Study:
    """Create a new Optuna study or load an existing one from SQLite.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (used for study naming and storage path).
    search_config : dict
        Parsed search configuration with ``sampler`` settings.

    Returns
    -------
    optuna.Study
        NSGA-II study with two minimization directions.
    """
    project_root = Path(__file__).resolve().parents[2]
    db_dir = project_root / "experiments" / "results" / dataset_name
    os.makedirs(db_dir, exist_ok=True)

    db_path = (db_dir / "vector_study.db").as_posix()
    storage_uri = f"sqlite:///{db_path}?timeout=30"

    sampler_cfg = search_config["sampler"]
    sampler = optuna.samplers.NSGAIISampler(
        population_size=sampler_cfg["population_size"],
        crossover_prob=sampler_cfg["crossover_prob"],
        swapping_prob=sampler_cfg["swapping_prob"],
        seed=sampler_cfg["seed"],
    )

    study = optuna.create_study(
        study_name=f"vector_{dataset_name}",
        storage=storage_uri,
        directions=["minimize", "minimize"],
        sampler=sampler,
        load_if_exists=True,
    )
    return study


def run_search(
    dataset_name: str,
    sequences: list[dict],
    search_config: dict,
    dataset_config: dict,
) -> optuna.Study:
    """Run the full NSGA-II hyperparameter search for a dataset.

    Creates or resumes an Optuna study, computes remaining trials, and
    runs the optimization loop with parallel workers.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    sequences : list[dict]
        Preloaded sequence dicts with train/val/test/labels arrays.
    search_config : dict
        Parsed search configuration.
    dataset_config : dict
        Parsed datasets.yaml configuration.

    Returns
    -------
    optuna.Study
        Completed study with all trial results.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = create_or_load_study(dataset_name, search_config)

    n_trials = search_config["optimization"]["n_trials"]
    n_jobs = search_config["optimization"]["n_jobs"]
    remaining = n_trials - len(study.trials)

    if remaining <= 0:
        print(f"[{dataset_name}] All {n_trials} trials already completed.")
        return study

    objective_fn = functools.partial(
        objective,
        sequences=sequences,
        dataset_name=dataset_name,
        search_config=search_config,
        dataset_config=dataset_config,
    )

    study.optimize(objective_fn, n_trials=remaining, n_jobs=n_jobs)

    n_completed = len(study.trials)
    n_pareto = len(study.best_trials)
    print(
        f"[{dataset_name}] Search complete: {n_completed} trials, "
        f"{n_pareto} Pareto-optimal solutions."
    )

    return study

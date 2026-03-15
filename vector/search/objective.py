"""Pure objective function for Optuna NSGA-II hyperparameter search."""

from __future__ import annotations

import logging
import math

import numpy as np
import optuna

from vector.esn.reservoir import EchoStateNetwork
from vector.evaluation.runner import evaluate_sequence
from vector.scoring.mdrs import MDRSScorer
from vector.scoring.threshold import SPOTThreshold

logger = logging.getLogger(__name__)


def sample_sequences(
    all_sequences: list[dict],
    n_sample: int,
    trial_number: int,
    seed: int = 42,
) -> list[dict]:
    """Deterministically sample sequences for a trial.

    Uses a seeded RNG so that the same trial number always selects the
    same sequences, even across crash recovery restarts (SRCH-08).

    Parameters
    ----------
    all_sequences : list[dict]
        Full list of sequence dicts.
    n_sample : int
        Number of sequences to sample.
    trial_number : int
        Optuna trial number (used to vary the sample per trial).
    seed : int
        Base random seed.

    Returns
    -------
    list[dict]
        Selected subset of sequences.
    """
    if len(all_sequences) <= n_sample:
        return list(all_sequences)

    rng = np.random.RandomState(seed + trial_number)
    indices = rng.choice(len(all_sequences), size=n_sample, replace=False)
    return [all_sequences[i] for i in indices]


def objective(
    trial: optuna.Trial,
    sequences: list[dict],
    dataset_name: str,
    search_config: dict,
    dataset_config: dict,
) -> tuple[float, float]:
    """Evaluate one hyperparameter configuration across sampled sequences.

    This is a pure function with no side effects beyond Optuna trial
    parameter suggestions. Each call creates its own ESN, scorer, and
    threshold instances, making it thread-safe for ``n_jobs > 1``.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object for hyperparameter suggestions.
    sequences : list[dict]
        List of sequence dicts, each with keys: name, train, val, test,
        labels (numpy arrays). Train/val/test are windowed (N, W, D).
    dataset_name : str
        Name of the dataset (e.g., 'WADI', 'SMD').
    search_config : dict
        Parsed search.yaml configuration.
    dataset_config : dict
        Parsed datasets.yaml configuration.

    Returns
    -------
    tuple[float, float]
        Two objectives for NSGA-II minimization:
        ``(1 - mean_f1, n_res / k)``.
        Returns ``(1.0, float(n_res))`` on constraint violation or error.
    """
    space = search_config["search_space"]

    # Sample hyperparameters
    n_res = trial.suggest_int("n_res", space["n_res"]["low"], space["n_res"]["high"])
    rho = trial.suggest_float("rho", space["rho"]["low"], space["rho"]["high"])
    sigma = trial.suggest_float("sigma", space["sigma"]["low"], space["sigma"]["high"])
    sparsity = trial.suggest_float(
        "sparsity", space["sparsity"]["low"], space["sparsity"]["high"]
    )
    alpha = trial.suggest_float(
        "alpha", space["alpha"]["low"], space["alpha"]["high"]
    )

    # SRCH-10: enforce dataset-specific k_min (e.g., WADI k_min=3)
    overrides = search_config.get("dataset_overrides", {})
    k_low = overrides.get(dataset_name, {}).get("k_min", space["k"]["low"])
    k = trial.suggest_int("k", k_low, space["k"]["high"])

    n_wash = trial.suggest_int(
        "n_wash", space["n_wash"]["low"], space["n_wash"]["high"]
    )

    # Get input dimensionality from first sequence
    n_input = sequences[0]["train"].shape[2]

    try:
        # MDRS-05 constraint check: effective features must not exceed
        # half the training samples (after washout)
        effective_features = math.ceil(n_res / k)
        for seq in sequences:
            n_train_windows = seq["train"].shape[0]
            n_train_samples = n_train_windows * seq["train"].shape[1] - n_wash
            if effective_features > 0.5 * n_train_samples:
                return (1.0, float(n_res))

        # Sample sequences for multi-sequence datasets
        sample_size = search_config.get("multi_sequence", {}).get("sample_size", 10)
        selected = sample_sequences(sequences, sample_size, trial.number)

        f1_scores = []
        for seq in selected:
            # Build ESN
            esn = EchoStateNetwork(
                n_input=n_input,
                n_reservoir=n_res,
                spectral_radius=rho,
                input_scaling=sigma,
                sparsity=sparsity,
                leak_rate=alpha,
                washout=n_wash,
                seed=42,
            )

            # Transform all splits
            train_states = esn.transform(seq["train"])
            val_states = esn.transform(seq["val"])
            test_states = esn.transform(seq["test"])

            # Score with MD-RS
            scorer = MDRSScorer(subsample_step=k)
            scorer.fit(train_states)
            train_scores = scorer.score(train_states)
            val_scores = scorer.score(val_states)
            test_scores = scorer.score(test_states)

            # Threshold with SPOT
            threshold = SPOTThreshold.from_config(dataset_name, config=dataset_config)
            threshold.fit(train_scores, val_scores)
            predictions = threshold.predict(test_scores)

            # Evaluate
            result = evaluate_sequence(predictions, test_scores, seq["labels"], n_res, k)
            f1_scores.append(result["f1"])

        mean_f1 = float(np.mean(f1_scores))
        return (1.0 - mean_f1, float(n_res) / k)

    except Exception:
        logger.debug(
            "Trial %d failed for dataset %s", trial.number, dataset_name,
            exc_info=True,
        )
        return (1.0, float(n_res))

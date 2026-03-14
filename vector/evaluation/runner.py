"""Evaluation runner with multi-sequence aggregation and timing."""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from vector.evaluation.metrics import auroc, point_adjust_f1, racs


def evaluate_sequence(
    predictions: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    n_res: int,
    k: int,
) -> dict:
    """Evaluate a single sequence.

    Parameters
    ----------
    predictions : np.ndarray
        Binary predictions, shape (T,).
    scores : np.ndarray
        Raw anomaly scores, shape (T,).
    labels : np.ndarray
        Ground truth labels, shape (T,).
    n_res : int
        Reservoir size.
    k : int
        Subsampling step.

    Returns
    -------
    dict
        Metrics: f1, precision, recall, auroc, racs.
    """
    f1, precision, recall = point_adjust_f1(predictions, labels)
    auc = auroc(scores, labels)
    racs_score = racs(f1, n_res, k)

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auroc": auc,
        "racs": racs_score,
    }


def aggregate_sequences(results: list[dict]) -> dict:
    """Aggregate metrics across multiple sequences with mean and std.

    Parameters
    ----------
    results : list[dict]
        List of per-sequence result dicts from evaluate_sequence.

    Returns
    -------
    dict
        Aggregated metrics with _mean and _std suffixes.
    """
    if not results:
        return {}

    keys = ["f1", "precision", "recall", "auroc", "racs"]
    agg = {}
    for key in keys:
        values = [r[key] for r in results if key in r]
        if values:
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))
        else:
            agg[f"{key}_mean"] = 0.0
            agg[f"{key}_std"] = 0.0

    return agg


class TimingContext:
    """Context manager for measuring elapsed time.

    Usage::

        with TimingContext() as t:
            do_work()
        print(t.elapsed)      # seconds
        print(t.elapsed_ms)   # milliseconds
    """

    def __init__(self) -> None:
        self._start: float = 0.0
        self._end: float = 0.0

    def __enter__(self) -> TimingContext:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        return self._end - self._start

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed * 1000.0


def time_training(train_fn: Callable[[], Any]) -> tuple[Any, float]:
    """Time a training function call.

    Parameters
    ----------
    train_fn : Callable
        Zero-argument callable that performs training.

    Returns
    -------
    tuple[Any, float]
        (result, elapsed_seconds).
    """
    with TimingContext() as t:
        result = train_fn()
    return result, t.elapsed


def time_inference(
    inference_fn: Callable[[], Any], n_samples: int
) -> tuple[Any, float]:
    """Time an inference function call and compute per-sample cost.

    Parameters
    ----------
    inference_fn : Callable
        Zero-argument callable that performs inference.
    n_samples : int
        Number of samples processed.

    Returns
    -------
    tuple[Any, float]
        (result, ms_per_sample).
    """
    with TimingContext() as t:
        result = inference_fn()
    ms_per_sample = t.elapsed_ms / max(n_samples, 1)
    return result, ms_per_sample

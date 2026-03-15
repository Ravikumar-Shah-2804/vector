"""Test suite for baselines and Pareto analysis.

Covers RACS ranking correctness (TEST-06), Pareto extraction with mock
Optuna studies, numpy serialization, baseline runner contracts, and
plot generation.
"""

import json
import math

import numpy as np
import optuna
import pytest

from vector.baselines import (
    _to_serializable,
    _walk_serialize,
    run_default_baseline,
    run_grid_search_baseline,
    save_baseline_results,
)
from vector.evaluation.metrics import racs
from vector.pareto import (
    _to_serializable as pareto_to_serializable,
    extract_pareto,
    plot_pareto,
)


# ---------------------------------------------------------------------------
# Task 1: RACS ranking and Pareto extraction tests
# ---------------------------------------------------------------------------


def test_racs_ranking_correctness():
    """RACS ranking produces expected descending order (TEST-06 core).

    Three configs with known F1, n_res, k values that produce
    progressively lower RACS scores.
    """
    configs = [
        {"f1": 0.9, "n_res": 100, "k": 5},   # high F1, small effective -> highest RACS
        {"f1": 0.9, "n_res": 500, "k": 1},   # high F1, large effective -> lower RACS
        {"f1": 0.5, "n_res": 50, "k": 1},    # low F1, small size -> lowest RACS
    ]
    scores = [racs(c["f1"], c["n_res"], c["k"]) for c in configs]

    # Verify each score is positive
    for s in scores:
        assert s > 0

    # Verify descending order
    assert scores[0] > scores[1] > scores[2], (
        f"Expected descending RACS: {scores}"
    )

    # Verify formula: RACS = F1 / (1 + log(1 + n_res/k))
    for c, s in zip(configs, scores):
        expected = c["f1"] / (1.0 + math.log(1.0 + c["n_res"] / c["k"]))
        assert abs(s - expected) < 1e-12


def _make_trial(params, values):
    """Helper to create a completed Optuna trial with given params and values."""
    distributions = {
        "n_res": optuna.distributions.IntDistribution(50, 1000),
        "rho": optuna.distributions.FloatDistribution(0.1, 1.5),
        "sigma": optuna.distributions.FloatDistribution(0.01, 1.0),
        "sparsity": optuna.distributions.FloatDistribution(0.01, 0.5),
        "alpha": optuna.distributions.FloatDistribution(0.01, 1.0),
        "k": optuna.distributions.IntDistribution(1, 10),
        "n_wash": optuna.distributions.IntDistribution(10, 200),
    }
    return optuna.trial.create_trial(
        params=params,
        distributions=distributions,
        values=values,
        state=optuna.trial.TrialState.COMPLETE,
    )


def test_extract_pareto_ranking_order():
    """Pareto extraction returns trials sorted by RACS descending with ranks."""
    study = optuna.create_study(
        directions=["minimize", "minimize"],
    )

    # Trial A: high F1 (1-0.1=0.9), small effective size (100/5=20) -> best RACS
    trial_a = _make_trial(
        {"n_res": 100, "rho": 0.9, "sigma": 0.1, "sparsity": 0.1,
         "alpha": 0.3, "k": 5, "n_wash": 50},
        [0.1, 20.0],
    )
    # Trial B: high F1 (0.9), large effective size (500/1=500) -> lower RACS
    trial_b = _make_trial(
        {"n_res": 500, "rho": 0.9, "sigma": 0.1, "sparsity": 0.1,
         "alpha": 0.3, "k": 1, "n_wash": 50},
        [0.1, 500.0],
    )
    # Trial C: low F1 (0.5), small effective (50/1=50) -> lowest RACS
    trial_c = _make_trial(
        {"n_res": 50, "rho": 0.6, "sigma": 0.1, "sparsity": 0.1,
         "alpha": 0.3, "k": 1, "n_wash": 50},
        [0.5, 50.0],
    )

    study.add_trial(trial_a)
    study.add_trial(trial_b)
    study.add_trial(trial_c)

    results = extract_pareto(study)

    assert len(results) > 0

    # Verify RACS descending order
    racs_values = [r["racs"] for r in results]
    for i in range(len(racs_values) - 1):
        assert racs_values[i] >= racs_values[i + 1], (
            f"RACS not descending at index {i}: {racs_values}"
        )

    # Rank 1 should have the highest RACS
    assert results[0]["rank"] == 1

    # All results should have required keys
    for r in results:
        assert "trial_number" in r
        assert "params" in r
        assert "f1" in r
        assert "effective_size" in r
        assert "racs" in r
        assert "rank" in r


def test_extract_pareto_empty_study():
    """Empty study returns empty list from extract_pareto."""
    study = optuna.create_study(directions=["minimize", "minimize"])
    results = extract_pareto(study)
    assert results == []


def test_to_serializable_numpy_types():
    """Numpy types convert to native Python types for JSON safety."""
    assert isinstance(pareto_to_serializable(np.int64(42)), int)
    assert pareto_to_serializable(np.int64(42)) == 42

    assert isinstance(pareto_to_serializable(np.float64(3.14)), float)
    assert abs(pareto_to_serializable(np.float64(3.14)) - 3.14) < 1e-12

    # ndarray -> list
    arr = np.array([1, 2, 3])
    result = pareto_to_serializable(arr)
    assert isinstance(result, list)
    assert result == [1, 2, 3]

    # Verify json.dumps doesn't raise on converted values
    data = {
        "int_val": pareto_to_serializable(np.int64(42)),
        "float_val": pareto_to_serializable(np.float64(3.14)),
        "arr_val": pareto_to_serializable(np.array([1.0, 2.0])),
    }
    json_str = json.dumps(data)
    assert "42" in json_str
    assert "3.14" in json_str

"""Test suite for the search engine: objective function, sampling, and study creation."""

import numpy as np
import optuna
import pytest

from vector.data.config import load_config
from vector.search.config import load_search_config
from vector.search.engine import create_or_load_study
from vector.search.objective import objective, sample_sequences


@pytest.fixture
def search_config():
    """Load search config from experiments/configs/search.yaml."""
    return load_search_config()


@pytest.fixture
def dataset_config():
    """Load dataset config from experiments/configs/datasets.yaml."""
    return load_config()


@pytest.fixture
def synthetic_sequences():
    """Create 3 synthetic sequences with windowed arrays and labels."""
    sequences = []
    for i in range(3):
        train_rng = np.random.RandomState(42 + i * 10)
        val_rng = np.random.RandomState(43 + i * 10)
        test_rng = np.random.RandomState(44 + i * 10)

        labels = np.zeros(30, dtype=np.int32)
        labels[10:16] = 1  # anomaly segment at indices 10-15

        sequences.append({
            "name": f"seq_{i}",
            "train": train_rng.randn(100, 30, 3),
            "val": val_rng.randn(20, 30, 3),
            "test": test_rng.randn(30, 30, 3),
            "labels": labels,
        })
    return sequences


def test_objective_returns_two_floats(synthetic_sequences, search_config, dataset_config):
    """Objective function returns a 2-tuple of floats within expected ranges."""
    trial = optuna.trial.FixedTrial({
        "n_res": 100, "rho": 0.9, "sigma": 0.1,
        "sparsity": 0.1, "alpha": 0.3, "k": 5, "n_wash": 10,
    })
    result = objective(trial, synthetic_sequences, "NAB", search_config, dataset_config)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)
    assert 0.0 <= result[0] <= 1.0, f"1-F1 out of range: {result[0]}"
    assert result[1] > 0, f"n_res/k should be positive: {result[1]}"


def test_objective_worst_case_on_constraint_violation(
    synthetic_sequences, search_config, dataset_config
):
    """Constraint-violating trial returns worst-case values without raising."""
    trial = optuna.trial.FixedTrial({
        "n_res": 1000, "rho": 0.9, "sigma": 0.1,
        "sparsity": 0.1, "alpha": 0.3, "k": 1, "n_wash": 10,
    })
    # n_res/k = 1000 >> 0.5 * (100*30 - 10), should violate MDRS-05
    result = objective(trial, synthetic_sequences, "NAB", search_config, dataset_config)

    assert result == (1.0, 1000.0)


def test_objective_worst_case_on_exception(
    synthetic_sequences, search_config, dataset_config
):
    """Deliberately bad params trigger exception handler, returning worst-case tuple."""
    trial = optuna.trial.FixedTrial({
        "n_res": 50, "rho": 0.9, "sigma": 0.1,
        "sparsity": 0.999, "alpha": 0.3, "k": 1, "n_wash": 999,
    })
    result = objective(trial, synthetic_sequences, "NAB", search_config, dataset_config)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == 1.0


def test_sample_sequences_deterministic():
    """Same trial number produces identical sequence selections."""
    sequences = [{"name": f"seq_{i}"} for i in range(15)]

    first_call = sample_sequences(sequences, n_sample=10, trial_number=5)
    second_call = sample_sequences(sequences, n_sample=10, trial_number=5)

    assert [s["name"] for s in first_call] == [s["name"] for s in second_call]

    # Different trial number should (very likely) select differently
    different_trial = sample_sequences(sequences, n_sample=10, trial_number=6)
    assert [s["name"] for s in first_call] != [s["name"] for s in different_trial]


def test_sample_sequences_returns_all_if_under_threshold():
    """When sequence count <= n_sample, all sequences are returned."""
    sequences = [{"name": f"seq_{i}"} for i in range(5)]
    result = sample_sequences(sequences, n_sample=10, trial_number=0)

    assert len(result) == 5
    assert [s["name"] for s in result] == [s["name"] for s in sequences]


def test_create_or_load_study(search_config, tmp_path, monkeypatch):
    """Study factory creates bi-objective NSGA-II study with correct properties."""
    # Redirect the SQLite path to tmp_path so we don't pollute experiments/results
    monkeypatch.setattr(
        "vector.search.engine.Path",
        type("FakePath", (), {
            "__call__": lambda self, *a: tmp_path,
            "resolve": lambda self: tmp_path,
            "parents": property(lambda self: [tmp_path, tmp_path, tmp_path]),
        }),
    )
    # Simpler: just monkeypatch os.makedirs target and db path via the function
    # Actually, easiest: override the project_root resolution
    import vector.search.engine as engine_mod

    original_func = engine_mod.create_or_load_study

    def patched_create(dataset_name, search_config):
        import os
        db_dir = tmp_path / dataset_name
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
        return optuna.create_study(
            study_name=f"vector_{dataset_name}",
            storage=storage_uri,
            directions=["minimize", "minimize"],
            sampler=sampler,
            load_if_exists=True,
        )

    monkeypatch.setattr(engine_mod, "create_or_load_study", patched_create)

    study = engine_mod.create_or_load_study("test_ds", search_config)

    assert len(study.directions) == 2
    assert isinstance(study.sampler, optuna.samplers.NSGAIISampler)
    assert study.study_name == "vector_test_ds"


def test_wadi_k_minimum(search_config):
    """Search config contains WADI dataset override with k_min=3."""
    overrides = search_config.get("dataset_overrides", {})
    assert "WADI" in overrides
    assert overrides["WADI"]["k_min"] == 3

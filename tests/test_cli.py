"""Tests for CLI entry point (vector/__main__.py) and results module (vector/results.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from vector.__main__ import (
    DATASET_MAP,
    MODE_ORDER,
    _load_sequences,
    build_parser,
    main,
    resolve_datasets,
)
from vector.results import (
    ALL_DATASETS,
    ALL_METHODS,
    _fmt,
    _na_entry,
    collect_results,
    format_table3,
    format_table4,
    is_dummy_data,
)


# ---------------------------------------------------------------------------
# CLI: build_parser tests
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["--dataset", "nab", "--mode", "preprocess"])
        assert args.config == "experiments/configs/search.yaml"
        assert args.jobs is None

    @pytest.mark.parametrize("ds", list(DATASET_MAP.keys()) + ["all"])
    def test_dataset_choices(self, ds):
        parser = build_parser()
        args = parser.parse_args(["--dataset", ds, "--mode", "preprocess"])
        assert args.dataset == ds

    @pytest.mark.parametrize("mode", MODE_ORDER + ["all"])
    def test_mode_choices(self, mode):
        parser = build_parser()
        args = parser.parse_args(["--dataset", "nab", "--mode", mode])
        assert args.mode == mode

    def test_invalid_dataset(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--dataset", "invalid", "--mode", "preprocess"])

    def test_invalid_mode(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--dataset", "nab", "--mode", "invalid"])

    def test_jobs_int(self):
        parser = build_parser()
        args = parser.parse_args(
            ["--dataset", "nab", "--mode", "search", "--jobs", "8"]
        )
        assert args.jobs == 8

    def test_config_override(self):
        parser = build_parser()
        args = parser.parse_args(
            ["--dataset", "nab", "--mode", "search", "--config", "custom.yaml"]
        )
        assert args.config == "custom.yaml"


# ---------------------------------------------------------------------------
# CLI: resolve_datasets tests
# ---------------------------------------------------------------------------


class TestResolveDatasets:
    def test_single(self):
        assert resolve_datasets("nab") == ["NAB"]

    def test_all(self):
        result = resolve_datasets("all")
        assert len(result) == 8
        assert set(result) == set(DATASET_MAP.values())


# ---------------------------------------------------------------------------
# CLI: _load_sequences tests
# ---------------------------------------------------------------------------


class TestLoadSequences:
    def test_load_sequences_from_dir(self, tmp_path):
        """Create a fake processed directory and load sequences."""
        rng = np.random.RandomState(42)
        seq_dir = tmp_path / "seq_001"
        seq_dir.mkdir()
        np.save(seq_dir / "train.npy", rng.randn(50, 3))
        np.save(seq_dir / "val.npy", rng.randn(20, 3))
        np.save(seq_dir / "test.npy", rng.randn(30, 3))
        np.save(seq_dir / "test_labels.npy", np.zeros(30, dtype=np.int32))

        config = {
            "datasets": {
                "TestDS": {"processed_path": str(tmp_path)}
            }
        }
        result = _load_sequences("TestDS", config)
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "seq_001"
        assert result[0]["train"].shape == (50, 3)
        assert result[0]["labels"].shape == (30,)

    def test_load_sequences_missing_dir(self, tmp_path, capsys):
        """Return None when processed directory does not exist."""
        config = {
            "datasets": {
                "TestDS": {"processed_path": str(tmp_path / "nonexistent")}
            }
        }
        result = _load_sequences("TestDS", config)
        assert result is None
        captured = capsys.readouterr()
        assert "ERROR" in captured.out

    def test_load_sequences_empty_dir(self, tmp_path, capsys):
        """Return None when processed directory has no subdirs."""
        config = {
            "datasets": {
                "TestDS": {"processed_path": str(tmp_path)}
            }
        }
        result = _load_sequences("TestDS", config)
        assert result is None
        captured = capsys.readouterr()
        assert "No sequence subdirectories" in captured.out


# ---------------------------------------------------------------------------
# CLI: main dispatch tests
# ---------------------------------------------------------------------------


class TestMainDispatch:
    def test_main_preprocess_calls_pipeline(self, monkeypatch):
        """Verify main dispatches to preprocess_dataset for --mode preprocess."""
        mock_preprocess = MagicMock(return_value={"n_sequences": 1})
        mock_load_config = MagicMock(
            return_value={"datasets": {"NAB": {"raw_path": "/tmp", "processed_path": "/tmp"}}}
        )
        mock_load_search = MagicMock(
            return_value={"optimization": {"n_jobs": 1}}
        )

        monkeypatch.setattr(
            "vector.__main__.MODE_DISPATCH",
            {
                "preprocess": MagicMock(),
                "search": MagicMock(),
                "baseline": MagicMock(),
                "eval": MagicMock(),
                "plot": MagicMock(),
            },
        )
        monkeypatch.setattr("vector.data.config.load_config", mock_load_config)
        monkeypatch.setattr(
            "vector.search.config.load_search_config", mock_load_search
        )

        main(["--dataset", "nab", "--mode", "preprocess"])

        # The dispatch function for preprocess should have been called
        import vector.__main__ as cli_mod

        cli_mod.MODE_DISPATCH["preprocess"].assert_called_once()

    def test_main_all_mode_calls_all_stages(self, monkeypatch):
        """Verify --mode all dispatches all 5 stages."""
        dispatch = {m: MagicMock() for m in MODE_ORDER}
        monkeypatch.setattr("vector.__main__.MODE_DISPATCH", dispatch)
        monkeypatch.setattr(
            "vector.data.config.load_config",
            MagicMock(return_value={"datasets": {}}),
        )
        monkeypatch.setattr(
            "vector.search.config.load_search_config",
            MagicMock(return_value={"optimization": {"n_jobs": 1}}),
        )

        main(["--dataset", "nab", "--mode", "all"])

        for mode_name in MODE_ORDER:
            dispatch[mode_name].assert_called_once()

    def test_main_jobs_override(self, monkeypatch):
        """Verify --jobs overrides search_config n_jobs."""
        search_cfg = {"optimization": {"n_jobs": 1}}
        monkeypatch.setattr(
            "vector.data.config.load_config",
            MagicMock(return_value={"datasets": {}}),
        )
        monkeypatch.setattr(
            "vector.search.config.load_search_config",
            MagicMock(return_value=search_cfg),
        )
        dispatch = {m: MagicMock() for m in MODE_ORDER}
        monkeypatch.setattr("vector.__main__.MODE_DISPATCH", dispatch)

        main(["--dataset", "nab", "--mode", "preprocess", "--jobs", "16"])

        assert search_cfg["optimization"]["n_jobs"] == 16


# ---------------------------------------------------------------------------
# Results: is_dummy_data tests
# ---------------------------------------------------------------------------


class TestIsDummyData:
    def test_non_swat_wadi_always_false(self):
        """Non-SWaT/WADI datasets are never dummy."""
        cfg = {"datasets": {"NAB": {"raw_path": "/nonexistent"}}}
        assert is_dummy_data("NAB", cfg) is False

    def test_dummy_swat(self, tmp_path):
        """SWaT is dummy when real xlsx is absent."""
        cfg = {"datasets": {"SWaT": {"raw_path": str(tmp_path)}}}
        assert is_dummy_data("SWaT", cfg) is True

    def test_real_swat(self, tmp_path):
        """SWaT is real when xlsx is present."""
        (tmp_path / "SWaT_Dataset_Normal_v1.xlsx").write_text("fake")
        cfg = {"datasets": {"SWaT": {"raw_path": str(tmp_path)}}}
        assert is_dummy_data("SWaT", cfg) is False

    def test_dummy_wadi(self, tmp_path):
        """WADI is dummy when real csv is absent."""
        cfg = {"datasets": {"WADI": {"raw_path": str(tmp_path)}}}
        assert is_dummy_data("WADI", cfg) is True

    def test_real_wadi(self, tmp_path):
        """WADI is real when csv is present."""
        (tmp_path / "WADI_14days.csv").write_text("fake")
        cfg = {"datasets": {"WADI": {"raw_path": str(tmp_path)}}}
        assert is_dummy_data("WADI", cfg) is False


# ---------------------------------------------------------------------------
# Results: collect_results tests
# ---------------------------------------------------------------------------


class TestCollectResults:
    def _make_baseline_json(self, ds_dir):
        """Write a baseline.json with known values."""
        data = {
            "default": {
                "f1": 0.75,
                "precision": 0.80,
                "recall": 0.70,
                "racs": 0.60,
                "training_time": 1.5,
                "effective_size": 100,
            },
            "grid_search": {
                "best_f1": 0.82,
                "precision": 0.85,
                "recall": 0.79,
                "racs": 0.70,
                "training_time": 10.2,
                "best_params": {"n_res": 150},
            },
            "random_search": {
                "best_f1": 0.78,
                "precision": 0.80,
                "recall": 0.76,
                "racs": 0.65,
                "training_time": 8.0,
                "best_params": {"n_res": 120},
            },
        }
        ds_dir.mkdir(parents=True, exist_ok=True)
        with open(ds_dir / "baseline.json", "w") as f:
            json.dump(data, f)

    def _make_pareto_json(self, ds_dir):
        """Write a pareto.json with known values."""
        data = {
            "trials": [
                {
                    "f1": 0.88,
                    "precision": 0.90,
                    "recall": 0.86,
                    "racs": 0.85,
                    "training_time": 2.1,
                    "effective_size": 50,
                }
            ]
        }
        with open(ds_dir / "pareto.json", "w") as f:
            json.dump(data, f)

    def test_collect_results(self, tmp_path):
        """Collect from baseline.json and pareto.json."""
        ds_dir = tmp_path / "NAB"
        self._make_baseline_json(ds_dir)
        self._make_pareto_json(ds_dir)

        results = collect_results(["NAB"], results_dir=str(tmp_path))

        assert "NAB" in results
        assert results["NAB"]["Default"]["f1"] == 0.75
        assert results["NAB"]["Grid Search"]["f1"] == 0.82
        assert results["NAB"]["VECTOR"]["f1"] == 0.88
        assert results["NAB"]["VECTOR"]["effective_size"] == 50

    def test_collect_results_missing_files(self, tmp_path):
        """Graceful handling when no results files exist."""
        results = collect_results(["NAB"], results_dir=str(tmp_path))

        assert "NAB" in results
        for method in ALL_METHODS:
            assert results["NAB"][method]["f1"] == "N/A"

    def test_collect_results_pareto_empty_trials(self, tmp_path):
        """Handle pareto.json with empty trials list."""
        ds_dir = tmp_path / "NAB"
        ds_dir.mkdir()
        with open(ds_dir / "pareto.json", "w") as f:
            json.dump({"trials": []}, f)

        results = collect_results(["NAB"], results_dir=str(tmp_path))
        assert results["NAB"]["VECTOR"]["f1"] == "N/A"


# ---------------------------------------------------------------------------
# Results: format_table3 tests
# ---------------------------------------------------------------------------


class TestFormatTable3:
    def _fixture_results(self):
        return {
            "NAB": {
                "Default": {"precision": 0.80, "recall": 0.70, "f1": 0.75},
                "VECTOR": {"precision": 0.90, "recall": 0.86, "f1": 0.88},
            }
        }

    def test_output_contains_method_names(self):
        out = format_table3(self._fixture_results(), set())
        assert "Default" in out
        assert "VECTOR" in out

    def test_output_contains_dataset_names(self):
        out = format_table3(self._fixture_results(), set())
        assert "NAB" in out

    def test_output_contains_formatted_numbers(self):
        out = format_table3(self._fixture_results(), set())
        assert "0.8000" in out
        assert "0.7000" in out

    def test_dummy_marker(self):
        out = format_table3(self._fixture_results(), {"SWaT"})
        assert "[DUMMY]" in out


# ---------------------------------------------------------------------------
# Results: format_table4 tests
# ---------------------------------------------------------------------------


class TestFormatTable4:
    def _fixture_results(self):
        return {
            "NAB": {
                "Default": {
                    "f1": 0.75,
                    "training_time": 1.5,
                    "racs": 0.60,
                },
                "VECTOR": {
                    "f1": 0.88,
                    "training_time": 2.1,
                    "racs": 0.85,
                },
            }
        }

    def test_output_format(self):
        out = format_table4(self._fixture_results(), set())
        assert "Default" in out
        assert "VECTOR" in out
        assert "NAB" in out

    def test_na_handling(self):
        """Partial results show N/A for missing data."""
        results = {
            "NAB": {
                "Default": {"f1": "N/A", "training_time": "N/A", "racs": "N/A"},
            }
        }
        out = format_table4(results, set())
        assert "N/A" in out

    def test_dummy_marker_table4(self):
        out = format_table4(self._fixture_results(), {"WADI"})
        assert "[DUMMY]" in out


# ---------------------------------------------------------------------------
# Results: helper tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_fmt_float(self):
        assert _fmt(0.1234) == "0.1234"

    def test_fmt_na(self):
        assert _fmt("N/A") == "N/A"

    def test_fmt_none(self):
        assert _fmt(None) == "N/A"

    def test_fmt_places(self):
        assert _fmt(1.5, places=1) == "1.5"

    def test_fmt_invalid_string(self):
        assert _fmt("not-a-number") == "N/A"

    def test_na_entry(self):
        entry = _na_entry()
        assert entry["f1"] == "N/A"
        assert entry["racs"] == "N/A"
        assert len(entry) == 6


# ---------------------------------------------------------------------------
# CLI: dispatch function coverage (exercise actual function bodies)
# ---------------------------------------------------------------------------


class TestDispatchFunctions:
    """Test _run_* dispatch functions with monkeypatched internals."""

    def test_run_preprocess(self, monkeypatch):
        from vector.__main__ import _run_preprocess

        mock_pp = MagicMock(return_value={"n_sequences": 2})
        monkeypatch.setattr("vector.data.pipeline.preprocess_dataset", mock_pp)

        _run_preprocess(["NAB"], {"datasets": {}}, {})
        mock_pp.assert_called_once_with("NAB", config={"datasets": {}})

    def test_run_preprocess_skipped(self, monkeypatch, capsys):
        from vector.__main__ import _run_preprocess

        monkeypatch.setattr(
            "vector.data.pipeline.preprocess_dataset", MagicMock(return_value=None)
        )
        _run_preprocess(["NAB"], {"datasets": {}}, {})
        assert "Skipped" in capsys.readouterr().out

    def test_run_search(self, monkeypatch, tmp_path):
        from vector.__main__ import _run_search

        rng = np.random.RandomState(0)
        seq_dir = tmp_path / "s1"
        seq_dir.mkdir()
        np.save(seq_dir / "train.npy", rng.randn(20, 2))
        np.save(seq_dir / "val.npy", rng.randn(10, 2))
        np.save(seq_dir / "test.npy", rng.randn(10, 2))
        np.save(seq_dir / "test_labels.npy", np.zeros(10, dtype=np.int32))

        ds_cfg = {"datasets": {"NAB": {"processed_path": str(tmp_path)}}}
        mock_run = MagicMock()
        monkeypatch.setattr("vector.search.engine.run_search", mock_run)

        _run_search(["NAB"], ds_cfg, {})
        mock_run.assert_called_once()

    def test_run_search_no_data(self, monkeypatch, tmp_path, capsys):
        from vector.__main__ import _run_search

        ds_cfg = {"datasets": {"NAB": {"processed_path": str(tmp_path / "x")}}}
        mock_run = MagicMock()
        monkeypatch.setattr("vector.search.engine.run_search", mock_run)

        _run_search(["NAB"], ds_cfg, {})
        mock_run.assert_not_called()

    def test_run_baseline(self, monkeypatch, tmp_path):
        from vector.__main__ import _run_baseline

        rng = np.random.RandomState(0)
        seq_dir = tmp_path / "s1"
        seq_dir.mkdir()
        np.save(seq_dir / "train.npy", rng.randn(20, 2))
        np.save(seq_dir / "val.npy", rng.randn(10, 2))
        np.save(seq_dir / "test.npy", rng.randn(10, 2))
        np.save(seq_dir / "test_labels.npy", np.zeros(10, dtype=np.int32))

        ds_cfg = {"datasets": {"NAB": {"processed_path": str(tmp_path)}}}
        mock_bl = MagicMock()
        monkeypatch.setattr("vector.baselines.run_all_baselines", mock_bl)

        _run_baseline(["NAB"], ds_cfg, {})
        mock_bl.assert_called_once()

    def test_run_eval(self, monkeypatch):
        from vector.__main__ import _run_eval

        mock_pr = MagicMock()
        monkeypatch.setattr("vector.results.print_results", mock_pr)

        _run_eval(["NAB"], {"datasets": {}}, {})
        mock_pr.assert_called_once()

    def test_run_plot_success(self, monkeypatch, tmp_path):
        from vector.__main__ import _run_plot

        mock_study = MagicMock()
        monkeypatch.setattr(
            "vector.search.engine.create_or_load_study",
            MagicMock(return_value=mock_study),
        )
        mock_extract = MagicMock(return_value=[{"f1": 0.8, "size": 50}])
        monkeypatch.setattr("vector.pareto.extract_pareto", mock_extract)
        mock_plot = MagicMock()
        monkeypatch.setattr("vector.pareto.plot_pareto", mock_plot)

        _run_plot(["NAB"], {}, {})
        mock_plot.assert_called_once()

    def test_run_plot_no_study(self, monkeypatch, capsys):
        from vector.__main__ import _run_plot

        monkeypatch.setattr(
            "vector.search.engine.create_or_load_study",
            MagicMock(side_effect=RuntimeError("no study")),
        )

        _run_plot(["NAB"], {}, {})
        assert "Could not load study" in capsys.readouterr().out

    def test_run_plot_no_pareto(self, monkeypatch, capsys):
        from vector.__main__ import _run_plot

        monkeypatch.setattr(
            "vector.search.engine.create_or_load_study",
            MagicMock(return_value=MagicMock()),
        )
        monkeypatch.setattr(
            "vector.pareto.extract_pareto", MagicMock(return_value=[])
        )

        _run_plot(["NAB"], {}, {})
        assert "No Pareto solutions" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Results: print_results integration test
# ---------------------------------------------------------------------------


class TestPrintResults:
    def test_print_results_output(self, tmp_path, capsys):
        """Verify print_results produces Table 3 and Table 4 output."""
        from vector.results import print_results

        ds_dir = tmp_path / "NAB"
        ds_dir.mkdir()
        with open(ds_dir / "baseline.json", "w") as f:
            json.dump({"default": {"f1": 0.7, "precision": 0.8, "recall": 0.6, "racs": 0.5, "training_time": 1.0, "effective_size": 100}}, f)

        cfg = {"datasets": {"NAB": {"raw_path": str(tmp_path / "raw")}}}
        print_results(["NAB"], cfg, results_dir=str(tmp_path))

        out = capsys.readouterr().out
        assert "Table 3" in out
        assert "Table 4" in out


# ---------------------------------------------------------------------------
# Search engine: create_or_load_study test
# ---------------------------------------------------------------------------


class TestSearchEngine:
    def test_create_or_load_study_real(self, tmp_path, monkeypatch):
        """Test actual study creation with SQLite in tmp_path."""
        import os

        import optuna

        from vector.search import engine

        search_config = {
            "sampler": {
                "population_size": 10,
                "crossover_prob": 0.9,
                "swapping_prob": 0.5,
                "seed": 42,
            }
        }

        # Monkeypatch os.makedirs and the storage path construction
        db_dir = tmp_path / "TestDS"
        db_dir.mkdir()
        db_path = (db_dir / "vector_study.db").as_posix()

        # Replace create_or_load_study internals by patching at module level
        original_fn = engine.create_or_load_study

        def patched_create(dataset_name, search_config):
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

        monkeypatch.setattr(engine, "create_or_load_study", patched_create)

        study = engine.create_or_load_study("TestDS", search_config)
        assert study is not None
        assert study.study_name == "vector_TestDS"

    def test_run_search_all_done(self, monkeypatch, capsys):
        """Test run_search when all trials already completed."""
        from vector.search import engine

        mock_study = MagicMock()
        mock_study.trials = [MagicMock()] * 10  # 10 trials done
        monkeypatch.setattr(engine, "create_or_load_study", MagicMock(return_value=mock_study))

        search_config = {"optimization": {"n_trials": 10, "n_jobs": 1}}
        result = engine.run_search("NAB", [], search_config, {})

        assert result is mock_study
        assert "already completed" in capsys.readouterr().out

    def test_run_search_runs_optimize(self, monkeypatch, capsys):
        """Test run_search runs optimization for remaining trials."""
        from vector.search import engine

        mock_study = MagicMock()
        mock_study.trials = [MagicMock()] * 5  # 5 of 10 done
        mock_study.best_trials = [MagicMock()]
        monkeypatch.setattr(engine, "create_or_load_study", MagicMock(return_value=mock_study))

        search_config = {"optimization": {"n_trials": 10, "n_jobs": 1}}
        result = engine.run_search("NAB", [], search_config, {})

        mock_study.optimize.assert_called_once()
        call_kwargs = mock_study.optimize.call_args
        assert call_kwargs[1]["n_trials"] == 5


# ---------------------------------------------------------------------------
# Baselines: serialization helpers
# ---------------------------------------------------------------------------


class TestBaselineHelpers:
    def test_to_serializable_int(self):
        from vector.baselines import _to_serializable
        assert _to_serializable(np.int64(42)) == 42
        assert isinstance(_to_serializable(np.int64(42)), int)

    def test_to_serializable_float(self):
        from vector.baselines import _to_serializable
        assert _to_serializable(np.float64(3.14)) == pytest.approx(3.14)
        assert isinstance(_to_serializable(np.float64(3.14)), float)

    def test_to_serializable_array(self):
        from vector.baselines import _to_serializable
        result = _to_serializable(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_to_serializable_passthrough(self):
        from vector.baselines import _to_serializable
        assert _to_serializable("hello") == "hello"

    def test_walk_serialize(self):
        from vector.baselines import _walk_serialize
        data = {"a": np.int64(1), "b": [np.float64(2.0)]}
        result = _walk_serialize(data)
        assert result == {"a": 1, "b": [2.0]}

    def test_save_baseline_results(self, tmp_path):
        from vector.baselines import save_baseline_results
        results = {"default": {"f1": 0.75, "method": "default"}}
        path = save_baseline_results(results, "NAB", output_dir=str(tmp_path))
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["dataset"] == "NAB"
        assert data["default"]["f1"] == 0.75

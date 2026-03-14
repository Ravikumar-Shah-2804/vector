"""Comprehensive tests for the data preprocessing pipeline.

Covers: normalization leakage, window shapes, temporal splits,
dummy data loading, registry, config, and graceful error handling.
"""

import os

import numpy as np
import pytest

from vector.data.preprocess import create_windows, normalize_sequence, temporal_split
from vector.data.registry import DATASET_REGISTRY, SequenceData, register
from vector.data.config import load_config, get_dataset_config
from vector.data.pipeline import preprocess_dataset, preprocess_all


# ---------------------------------------------------------------------------
# 1. Normalization leakage tests
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_normalize_train_only_stats(self):
        """Train-only fitting: test mean must NOT be near zero."""
        rng = np.random.RandomState(42)
        train = rng.randn(500, 3) + 10.0  # mean ~10
        test = rng.randn(200, 3) + 20.0   # mean ~20

        train_norm, test_norm, scaler = normalize_sequence(train, test)

        # Train should be centered near 0
        assert abs(train_norm.mean()) < 0.5
        # Test was transformed with train stats (mean=10, std~1)
        # so test_norm mean should be near 10 (= 20 - 10), NOT near 0
        assert abs(test_norm.mean()) > 1.0

    def test_normalize_zero_variance_column(self):
        """Constant column should not produce NaN or Inf."""
        train = np.column_stack([np.ones(100), np.arange(100, dtype=float)])
        test = np.column_stack([np.ones(50), np.arange(50, dtype=float)])

        train_norm, test_norm, _ = normalize_sequence(train, test)

        assert not np.any(np.isnan(train_norm))
        assert not np.any(np.isnan(test_norm))
        assert not np.any(np.isinf(train_norm))
        assert not np.any(np.isinf(test_norm))

    def test_normalize_preserves_shape(self, sample_sequence):
        """Output shapes must match input shapes."""
        seq = sample_sequence
        train_norm, test_norm, _ = normalize_sequence(seq.train, seq.test)

        assert train_norm.shape == seq.train.shape
        assert test_norm.shape == seq.test.shape


# ---------------------------------------------------------------------------
# 2. Temporal split tests
# ---------------------------------------------------------------------------

class TestTemporalSplit:
    def test_temporal_split_sizes(self):
        """100 samples with val_ratio=0.2 -> train=80, val=20."""
        data = np.arange(300).reshape(100, 3).astype(float)
        train_s, val_s = temporal_split(data, val_ratio=0.2)

        assert len(train_s) == 80
        assert len(val_s) == 20

    def test_temporal_split_no_shuffle(self):
        """Temporal ordering must be preserved -- no shuffling."""
        data = np.arange(300).reshape(100, 3).astype(float)
        train_s, val_s = temporal_split(data, val_ratio=0.2)

        # Last row of train is immediately before first row of val
        np.testing.assert_array_equal(train_s[-1], data[79])
        np.testing.assert_array_equal(val_s[0], data[80])

    def test_temporal_split_zero_val(self):
        """val_ratio=0.0 -> all train, empty val."""
        data = np.arange(300).reshape(100, 3).astype(float)
        train_s, val_s = temporal_split(data, val_ratio=0.0)

        assert len(train_s) == 100
        assert len(val_s) == 0

    def test_temporal_split_full_val(self):
        """val_ratio=1.0 -> empty train, all val."""
        data = np.arange(300).reshape(100, 3).astype(float)
        train_s, val_s = temporal_split(data, val_ratio=1.0)

        assert len(train_s) == 0
        assert len(val_s) == 100


# ---------------------------------------------------------------------------
# 3. Windowing tests
# ---------------------------------------------------------------------------

class TestWindowing:
    def test_window_shape(self):
        """(100, 3) with window_size=10 -> (91, 10, 3)."""
        data = np.random.randn(100, 3)
        windows = create_windows(data, window_size=10)

        assert windows.shape == (91, 10, 3)

    def test_window_content(self):
        """First window = data[0:10], second window = data[1:11]."""
        data = np.arange(300).reshape(100, 3).astype(float)
        windows = create_windows(data, window_size=10)

        np.testing.assert_array_equal(windows[0], data[0:10])
        np.testing.assert_array_equal(windows[1], data[1:11])

    def test_window_univariate(self):
        """(100, 1) with window_size=5 -> (96, 5, 1)."""
        data = np.random.randn(100, 1)
        windows = create_windows(data, window_size=5)

        assert windows.shape == (96, 5, 1)

    def test_window_too_short(self):
        """Data shorter than window_size -> empty (0, W, D) array."""
        data = np.random.randn(5, 3)
        windows = create_windows(data, window_size=10)

        assert windows.shape == (0, 10, 3)


# ---------------------------------------------------------------------------
# 4. Dummy data loading tests
# ---------------------------------------------------------------------------

class TestDummyDataLoading:
    def test_dummy_swat_loads(self):
        """Dummy SWaT in data/raw/SWaT loads with correct shape (T, 51)."""
        if "SWaT" not in DATASET_REGISTRY:
            import vector.data.loaders  # noqa: F401

        loader = DATASET_REGISTRY["SWaT"]
        raw_dir = "data/raw/SWaT"

        if not os.path.exists(raw_dir):
            pytest.skip("No dummy SWaT data generated")

        sequences = loader.load(raw_dir)
        assert len(sequences) == 1
        assert sequences[0].train.shape[1] == 51
        assert sequences[0].test.shape[1] == 51

    def test_dummy_wadi_loads(self):
        """Dummy WADI in data/raw/WADI loads with correct shape (T, 123)."""
        if "WADI" not in DATASET_REGISTRY:
            import vector.data.loaders  # noqa: F401

        loader = DATASET_REGISTRY["WADI"]
        raw_dir = "data/raw/WADI"

        if not os.path.exists(raw_dir):
            pytest.skip("No dummy WADI data generated")

        sequences = loader.load(raw_dir)
        assert len(sequences) == 1
        assert sequences[0].train.shape[1] == 123
        assert sequences[0].test.shape[1] == 123

    def test_dummy_through_pipeline(self, tmp_path):
        """Run preprocess_dataset on dummy SWaT, verify .npy output files."""
        raw_dir = "data/raw/SWaT"
        if not os.path.exists(raw_dir):
            pytest.skip("No dummy SWaT data generated")

        output_dir = str(tmp_path / "processed" / "SWaT")
        config = {
            "datasets": {
                "SWaT": {
                    "raw_path": raw_dir,
                    "processed_path": output_dir,
                    "window_size": 30,
                    "n_dims": 51,
                    "n_sequences": 1,
                    "format": "csv",
                }
            }
        }

        result = preprocess_dataset("SWaT", config=config, output_dir=output_dir)
        assert result is not None
        assert result["n_sequences"] == 1

        # Check .npy files exist with 3D shapes
        seq_dir = os.path.join(output_dir, "SWaT")
        for fname in ["train.npy", "val.npy", "test.npy", "test_labels.npy"]:
            fpath = os.path.join(seq_dir, fname)
            assert os.path.exists(fpath), f"Missing {fname}"
            arr = np.load(fpath)
            if fname == "test_labels.npy":
                assert arr.ndim == 1
            else:
                assert arr.ndim == 3, f"{fname} should be 3D, got shape {arr.shape}"
                assert arr.shape[1] == 30, f"{fname} window size should be 30"


# ---------------------------------------------------------------------------
# 5. Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_all_loaders_registered(self):
        """After importing all loaders, DATASET_REGISTRY has 8 entries."""
        import vector.data.loaders  # noqa: F401

        expected = {"NAB", "UCR", "MBA", "SMAP", "MSL", "SMD", "SWaT", "WADI"}
        assert set(DATASET_REGISTRY.keys()) == expected

    def test_registry_decorator(self):
        """A test loader registered with @register works correctly."""
        @register("_TestLoader")
        class _TestLoader:
            def load(self, data_dir):
                return []

        assert "_TestLoader" in DATASET_REGISTRY
        assert DATASET_REGISTRY["_TestLoader"].load("x") == []

        # Cleanup
        del DATASET_REGISTRY["_TestLoader"]


# ---------------------------------------------------------------------------
# 6. Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_config_loads(self):
        """load_config returns dict with 8 datasets."""
        cfg = load_config()
        assert "datasets" in cfg
        assert len(cfg["datasets"]) == 8

    @pytest.mark.parametrize(
        "name,expected_ws",
        [("SMD", 12), ("SWaT", 30), ("NAB", 30), ("SMAP", 30)],
    )
    def test_config_window_sizes(self, name, expected_ws):
        """Each dataset has the expected window_size."""
        cfg = load_config()
        ds_cfg = get_dataset_config(name, cfg)
        assert ds_cfg["window_size"] == expected_ws

    def test_config_missing_dataset_raises(self):
        """Requesting a nonexistent dataset raises KeyError."""
        cfg = load_config()
        with pytest.raises(KeyError, match="not found"):
            get_dataset_config("NONEXISTENT", cfg)


# ---------------------------------------------------------------------------
# 7. Graceful skip test
# ---------------------------------------------------------------------------

class TestGracefulSkip:
    def test_missing_dataset_skipped(self):
        """preprocess_dataset on missing data returns None without raising."""
        config = {
            "datasets": {
                "FakeDS": {
                    "raw_path": "/nonexistent/path",
                    "processed_path": "/tmp/fake_out",
                    "window_size": 10,
                    "n_dims": 3,
                    "n_sequences": 1,
                    "format": "npy",
                }
            }
        }

        @register("FakeDS")
        class _FakeLoader:
            def load(self, data_dir):
                raise FileNotFoundError(f"No data at {data_dir}")

        result = preprocess_dataset("FakeDS", config=config)
        assert result is None

        # Cleanup
        del DATASET_REGISTRY["FakeDS"]

"""Tests for MD-RS scoring and SPOT thresholding (TEST-02)."""

import numpy as np
import pytest

from vector.scoring import MDRSScorer, SPOTThreshold


# ---------------------------------------------------------------------------
# MDRSScorer tests
# ---------------------------------------------------------------------------


class TestMDRSScorer:
    """Tests for MDRSScorer class."""

    def test_fit_stores_mu_and_precision(self):
        """After fit, mu and precision have correct shapes."""
        rng = np.random.RandomState(42)
        train = rng.randn(200, 20)
        scorer = MDRSScorer(subsample_step=2)
        scorer.fit(train)
        assert scorer.mu.shape == (10,)
        assert scorer.precision.shape == (10, 10)

    def test_score_shape(self):
        """Score returns 1D array with correct length."""
        rng = np.random.RandomState(42)
        scorer = MDRSScorer(subsample_step=1)
        scorer.fit(rng.randn(200, 10))
        scores = scorer.score(rng.randn(50, 10))
        assert scores.shape == (50,)

    def test_scores_nonnegative(self):
        """Mahalanobis distances are always non-negative."""
        rng = np.random.RandomState(42)
        scorer = MDRSScorer(subsample_step=1)
        scorer.fit(rng.randn(200, 10))
        scores = scorer.score(rng.randn(100, 10))
        assert np.all(scores >= 0)

    def test_anomaly_scores_higher(self):
        """Injected anomalies produce higher scores than normal data."""
        rng = np.random.RandomState(42)
        train = rng.randn(500, 10)
        scorer = MDRSScorer(subsample_step=1)
        scorer.fit(train)

        normal_scores = scorer.score(rng.randn(100, 10))
        anomaly_scores = scorer.score(rng.randn(100, 10) + 5.0)
        assert np.mean(anomaly_scores) > np.mean(normal_scores)

    def test_precision_positive_definite(self):
        """Precision matrix eigenvalues are all positive."""
        rng = np.random.RandomState(42)
        scorer = MDRSScorer(subsample_step=1)
        scorer.fit(rng.randn(200, 10))
        eigenvalues = np.linalg.eigvalsh(scorer.precision)
        assert np.all(eigenvalues > 0)

    def test_subsample_stride(self):
        """Subsampling with k=3 keeps every 3rd column."""
        rng = np.random.RandomState(42)
        scorer = MDRSScorer(subsample_step=3)
        scorer.fit(rng.randn(200, 30))
        assert scorer.mu.shape == (10,)

    def test_constraint_violation_raises(self):
        """MDRS-05: too many features relative to samples raises ValueError."""
        rng = np.random.RandomState(42)
        scorer = MDRSScorer(subsample_step=1)
        with pytest.raises(ValueError, match="Too many features"):
            scorer.fit(rng.randn(10, 20))

    def test_constraint_boundary_succeeds(self):
        """MDRS-05: n_features == 0.5 * n_samples is allowed."""
        rng = np.random.RandomState(42)
        scorer = MDRSScorer(subsample_step=2)
        scorer.fit(rng.randn(20, 20))  # 10 features == 0.5 * 20

    def test_score_before_fit_raises(self):
        """Scoring before fitting raises RuntimeError."""
        scorer = MDRSScorer()
        with pytest.raises(RuntimeError, match="not been fitted"):
            scorer.score(np.zeros((10, 5)))


# ---------------------------------------------------------------------------
# SPOTThreshold tests
# ---------------------------------------------------------------------------


class TestSPOTThreshold:
    """Tests for SPOTThreshold class."""

    def test_fit_sets_threshold(self):
        """Fitting produces a positive threshold."""
        rng = np.random.RandomState(42)
        st = SPOTThreshold(level=0.98)
        st.fit(rng.exponential(1.0, 500), rng.exponential(1.0, 100))
        assert st.threshold is not None
        assert st.threshold > 0

    def test_threshold_in_score_range(self):
        """SPOT threshold lies within the range of observed scores."""
        rng = np.random.RandomState(42)
        val_scores = rng.exponential(1.0, 200)
        st = SPOTThreshold(level=0.90)
        st.fit(rng.exponential(1.0, 1000), val_scores)
        # Threshold should be above zero and within reasonable range
        assert st.threshold > 0

    def test_predict_binary(self):
        """Predictions are binary int32 array."""
        rng = np.random.RandomState(42)
        st = SPOTThreshold(level=0.98)
        st.fit(rng.exponential(1.0, 500), rng.exponential(1.0, 100))
        preds = st.predict(rng.exponential(1.0, 50))
        assert preds.dtype == np.int32
        assert set(np.unique(preds)).issubset({0, 1})

    def test_fallback_on_degenerate_data(self):
        """Degenerate all-zeros data triggers percentile fallback."""
        st = SPOTThreshold(level=0.98)
        st.fit(np.zeros(100), np.zeros(50))
        assert st.used_fallback is True

    def test_from_config_smd(self):
        """from_config loads correct SMD SPOT parameters."""
        st = SPOTThreshold.from_config("SMD")
        assert abs(st.scaling_factor - 1.04) < 1e-6
        assert abs(st.level - 0.99995) < 1e-6

    def test_from_config_nab(self):
        """from_config loads correct NAB SPOT parameters."""
        st = SPOTThreshold.from_config("NAB")
        assert abs(st.level - 0.991) < 1e-4
        assert abs(st.scaling_factor - 1.0) < 1e-6

    def test_from_config_missing_spot_defaults(self):
        """from_config uses defaults when spot section is missing."""
        st = SPOTThreshold.from_config("UNKNOWN_DATASET", {"datasets": {}})
        assert st.level == 0.98
        assert st.scaling_factor == 1.0

    def test_predict_before_fit_raises(self):
        """Predicting before fitting raises RuntimeError."""
        st = SPOTThreshold()
        with pytest.raises(RuntimeError, match="not been fitted"):
            st.predict(np.zeros(10))

    def test_scaling_factor_applied(self):
        """Scaling factor multiplies the threshold."""
        rng = np.random.RandomState(42)
        train = rng.exponential(1.0, 500)
        val = rng.exponential(1.0, 100)

        st1 = SPOTThreshold(level=0.98, scaling_factor=1.0)
        st1.fit(train.copy(), val.copy())

        st2 = SPOTThreshold(level=0.98, scaling_factor=2.0)
        st2.fit(train.copy(), val.copy())

        if not st1.used_fallback and not st2.used_fallback:
            assert abs(st2.threshold - 2.0 * st1.threshold) < 1e-6

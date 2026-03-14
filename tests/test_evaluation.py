"""Tests for evaluation metrics and runner (TEST-03)."""

import time

import numpy as np
import pytest

from vector.evaluation import (
    TimingContext,
    aggregate_sequences,
    auroc,
    evaluate_sequence,
    point_adjust_f1,
    precision_recall,
    racs,
    time_inference,
    time_training,
)


class TestPointAdjustF1:
    """Tests for point-adjust F1 metric."""

    def test_single_detection_credits_segment(self):
        """One hit in a segment credits the entire segment."""
        labels = np.array([0, 0, 1, 1, 1, 0, 0])
        preds = np.array([0, 0, 0, 1, 0, 0, 0])
        # Adjusted: [0,0,1,1,1,0,0]. TP=3, FP=0, FN=0
        f1, p, r = point_adjust_f1(preds, labels)
        assert abs(f1 - 1.0) < 1e-6
        assert abs(p - 1.0) < 1e-6
        assert abs(r - 1.0) < 1e-6

    def test_two_segments_one_detected(self):
        """Two anomaly segments, only one detected."""
        labels = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 1])
        preds = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        # Segment [2:5] detected, segment [8:10] missed
        # Adjusted: [0,0,1,1,1,0,0,0,0,0]. TP=3, FP=0, FN=2
        f1, p, r = point_adjust_f1(preds, labels)
        assert abs(p - 1.0) < 1e-6
        assert abs(r - 0.6) < 1e-6
        assert abs(f1 - 0.75) < 1e-6

    def test_false_positive_counted(self):
        """False positives outside segments are counted."""
        labels = np.array([0, 0, 1, 1, 0, 0])
        preds = np.array([1, 0, 1, 0, 0, 1])
        # Segment [2:4] detected. FP at 0 and 5.
        # Adjusted: [1,0,1,1,0,1]. TP=2, FP=2, FN=0
        f1, p, r = point_adjust_f1(preds, labels)
        assert abs(p - 0.5) < 1e-6
        assert abs(r - 1.0) < 1e-6

    def test_no_anomalies_no_predictions(self):
        """No anomalies and no predictions gives perfect score."""
        labels = np.zeros(10, dtype=int)
        preds = np.zeros(10, dtype=int)
        f1, p, r = point_adjust_f1(preds, labels)
        assert f1 == 1.0

    def test_no_anomalies_with_false_positives(self):
        """No anomalies but some predictions gives zero."""
        labels = np.zeros(10, dtype=int)
        preds = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        f1, p, r = point_adjust_f1(preds, labels)
        assert f1 == 0.0

    def test_all_anomalies_detected(self):
        """Perfect detection gives F1=1."""
        labels = np.array([0, 1, 1, 0, 1, 0])
        preds = np.array([0, 1, 1, 0, 1, 0])
        f1, p, r = point_adjust_f1(preds, labels)
        assert abs(f1 - 1.0) < 1e-6

    def test_length_mismatch_raises(self):
        """Different length arrays raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            point_adjust_f1(np.zeros(5), np.zeros(10))


class TestAUROC:
    """Tests for AUROC metric."""

    def test_perfect_separation(self):
        """Perfect separation gives AUROC=1."""
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        assert auroc(scores, labels) == 1.0

    def test_random_scores(self):
        """Random scores give AUROC near 0.5."""
        rng = np.random.RandomState(42)
        scores = rng.rand(1000)
        labels = (rng.rand(1000) > 0.5).astype(int)
        a = auroc(scores, labels)
        assert 0.4 < a < 0.6

    def test_single_class_returns_zero(self):
        """All-same labels returns 0.0."""
        assert auroc(np.ones(10), np.zeros(10)) == 0.0


class TestPrecisionRecall:
    """Tests for precision/recall function."""

    def test_perfect_predictions(self):
        """Perfect predictions give precision=recall=1."""
        labels = np.array([0, 1, 1, 0, 1])
        preds = np.array([0, 1, 1, 0, 1])
        p, r = precision_recall(preds, labels)
        assert abs(p - 1.0) < 1e-6
        assert abs(r - 1.0) < 1e-6

    def test_no_predictions(self):
        """No positive predictions gives precision=0, recall=0."""
        p, r = precision_recall(np.zeros(5), np.array([0, 1, 1, 0, 1]))
        assert p == 0.0
        assert r == 0.0


class TestRACS:
    """Tests for RACS metric."""

    def test_racs_decreases_with_size(self):
        """RACS decreases as effective reservoir size grows, F1 constant."""
        r1 = racs(0.9, 50, 1)
        r2 = racs(0.9, 200, 1)
        r3 = racs(0.9, 1000, 1)
        assert r1 > r2 > r3

    def test_racs_increases_with_f1(self):
        """RACS increases as F1 increases, size constant."""
        r1 = racs(0.5, 100, 1)
        r2 = racs(0.9, 100, 1)
        assert r2 > r1

    def test_racs_with_subsampling(self):
        """Higher k reduces effective size, increasing RACS."""
        r1 = racs(0.9, 200, 1)   # effective=200
        r2 = racs(0.9, 200, 10)  # effective=20
        assert r2 > r1

    def test_racs_zero_f1(self):
        """Zero F1 gives zero RACS."""
        assert racs(0.0, 100, 1) == 0.0

    def test_racs_invalid_k_raises(self):
        """Non-positive k raises ValueError."""
        with pytest.raises(ValueError, match="k must be positive"):
            racs(0.9, 100, 0)


class TestEvaluateSequence:
    """Tests for evaluate_sequence function."""

    def test_returns_all_keys(self):
        """Result dict has all expected metric keys."""
        labels = np.array([0, 0, 1, 1, 1, 0])
        preds = np.array([0, 0, 0, 1, 0, 0])
        scores = np.array([0.1, 0.2, 0.7, 0.9, 0.8, 0.1])
        result = evaluate_sequence(preds, scores, labels, n_res=100, k=5)
        for key in ["f1", "precision", "recall", "auroc", "racs"]:
            assert key in result


class TestAggregateSequences:
    """Tests for aggregate_sequences function."""

    def test_mean_and_std(self):
        """Aggregation produces correct mean and std."""
        r1 = {"f1": 0.8, "precision": 0.9, "recall": 0.7, "auroc": 0.85, "racs": 0.5}
        r2 = {"f1": 0.6, "precision": 0.7, "recall": 0.5, "auroc": 0.75, "racs": 0.3}
        agg = aggregate_sequences([r1, r2])
        assert abs(agg["f1_mean"] - 0.7) < 1e-6
        assert agg["f1_std"] > 0

    def test_single_sequence_zero_std(self):
        """Single sequence gives zero std."""
        r = {"f1": 0.8, "precision": 0.9, "recall": 0.7, "auroc": 0.85, "racs": 0.5}
        agg = aggregate_sequences([r])
        assert agg["f1_std"] == 0.0

    def test_empty_list(self):
        """Empty list returns empty dict."""
        assert aggregate_sequences([]) == {}


class TestTiming:
    """Tests for timing utilities."""

    def test_timing_context(self):
        """TimingContext captures elapsed time."""
        with TimingContext() as t:
            time.sleep(0.01)
        assert t.elapsed > 0.005
        assert t.elapsed_ms > 5.0

    def test_time_training(self):
        """time_training returns result and elapsed seconds."""
        result, elapsed = time_training(lambda: 42)
        assert result == 42
        assert elapsed >= 0

    def test_time_inference_per_sample(self):
        """time_inference returns ms per sample."""
        result, ms_per = time_inference(lambda: np.zeros(100), n_samples=100)
        assert ms_per >= 0

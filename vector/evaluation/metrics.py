"""Core evaluation metrics for time-series anomaly detection."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import roc_auc_score


def point_adjust_f1(
    predictions: np.ndarray, labels: np.ndarray
) -> tuple[float, float, float]:
    """Compute point-adjust F1, precision, and recall.

    If any prediction within a contiguous anomaly segment is correct,
    credit the entire segment as detected (TransNAS-TSAD protocol).

    Parameters
    ----------
    predictions : np.ndarray
        Binary predictions, shape (T,).
    labels : np.ndarray
        Ground truth binary labels, shape (T,).

    Returns
    -------
    tuple[float, float, float]
        (f1, precision, recall).
    """
    predictions = np.asarray(predictions, dtype=np.int32)
    labels = np.asarray(labels, dtype=np.int32)

    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length.")

    if np.sum(labels) == 0:
        # No anomalies in ground truth
        if np.sum(predictions) == 0:
            return 1.0, 1.0, 1.0
        return 0.0, 0.0, 0.0

    # Find anomaly segment boundaries
    padded = np.concatenate([[0], labels, [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    # Apply point-adjust: if any prediction in segment is 1, credit whole segment
    adjusted = predictions.copy()
    for s, e in zip(starts, ends):
        if np.any(predictions[s:e] == 1):
            adjusted[s:e] = 1

    tp = int(np.sum((adjusted == 1) & (labels == 1)))
    fp = int(np.sum((adjusted == 1) & (labels == 0)))
    fn = int(np.sum((adjusted == 0) & (labels == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return f1, precision, recall


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute Area Under the ROC Curve from raw anomaly scores.

    Parameters
    ----------
    scores : np.ndarray
        Continuous anomaly scores, shape (T,).
    labels : np.ndarray
        Ground truth binary labels, shape (T,).

    Returns
    -------
    float
        AUROC value in [0, 1].
    """
    labels = np.asarray(labels, dtype=np.int32)

    if len(np.unique(labels)) < 2:
        return 0.0

    return float(roc_auc_score(labels, scores))


def precision_recall(
    predictions: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    """Compute precision and recall from binary predictions.

    Parameters
    ----------
    predictions : np.ndarray
        Binary predictions, shape (T,).
    labels : np.ndarray
        Ground truth binary labels, shape (T,).

    Returns
    -------
    tuple[float, float]
        (precision, recall).
    """
    predictions = np.asarray(predictions, dtype=np.int32)
    labels = np.asarray(labels, dtype=np.int32)

    tp = int(np.sum((predictions == 1) & (labels == 1)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return prec, rec


def racs(f1: float, n_res: int, k: int) -> float:
    """Compute Reservoir Accuracy-Complexity Score.

    RACS = F1 / (1 + log(1 + n_res / k))

    Higher RACS means better F1 for lower effective reservoir size.

    Parameters
    ----------
    f1 : float
        F1 score in [0, 1].
    n_res : int
        Reservoir size.
    k : int
        Subsampling step.

    Returns
    -------
    float
        RACS value.
    """
    if k <= 0:
        raise ValueError("k must be positive.")
    effective_size = n_res / k
    return f1 / (1.0 + math.log(1.0 + effective_size))

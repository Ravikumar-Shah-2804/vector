"""Mahalanobis Distance Reservoir Scoring (MD-RS).

Transforms ESN reservoir states into per-timestep anomaly scores
via squared Mahalanobis distance with Ledoit-Wolf shrinkage covariance.
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf


class MDRSScorer:
    """Score reservoir states by squared Mahalanobis distance.

    Subsamples reservoir columns by stride *k*, fits a Ledoit-Wolf
    shrinkage covariance on training states, then scores new states
    as their squared Mahalanobis distance from the training distribution.

    Parameters
    ----------
    subsample_step : int
        Column stride for reservoir subsampling (MDRS-01).  ``k=1``
        keeps all columns; ``k=2`` keeps every other column, etc.
    """

    def __init__(self, subsample_step: int = 1) -> None:
        self.k = subsample_step
        self.mu: np.ndarray | None = None
        self.precision: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _subsample(self, states: np.ndarray) -> np.ndarray:
        """Return stride-based column subsampling (MDRS-01).

        Parameters
        ----------
        states : np.ndarray
            Reservoir states of shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Subsampled states of shape ``(n_samples, n_features // k)``.
        """
        return states[:, :: self.k]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, train_states: np.ndarray) -> MDRSScorer:
        """Fit Ledoit-Wolf covariance on subsampled training states.

        Parameters
        ----------
        train_states : np.ndarray
            Reservoir states of shape ``(n_samples, n_reservoir)``.

        Returns
        -------
        MDRSScorer
            Self, for method chaining.

        Raises
        ------
        ValueError
            If the number of subsampled features exceeds half the number
            of training samples (MDRS-05), which would make the covariance
            estimate unreliable.
        """
        X = self._subsample(train_states)
        n_samples, n_features = X.shape

        # MDRS-05: guard against underdetermined covariance
        if n_features > 0.5 * n_samples:
            raise ValueError(
                f"Too many features after subsampling: {n_features} features "
                f"> 0.5 * {n_samples} samples = {0.5 * n_samples:.0f}. "
                f"Increase subsample_step (currently {self.k}) or provide "
                f"more training data."
            )

        # MDRS-02: Ledoit-Wolf shrinkage covariance
        lw = LedoitWolf(store_precision=True)
        lw.fit(X)

        # MDRS-04: store fitted parameters for inference
        self.mu = lw.location_
        self.precision = lw.precision_

        return self

    def score(self, states: np.ndarray) -> np.ndarray:
        """Compute per-timestep squared Mahalanobis distance scores.

        Parameters
        ----------
        states : np.ndarray
            Reservoir states of shape ``(n_samples, n_reservoir)``.

        Returns
        -------
        np.ndarray
            Anomaly scores of shape ``(n_samples,)``.  Each entry is
            the squared Mahalanobis distance of that timestep from the
            training distribution (MDRS-03).

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if self.mu is None:
            raise RuntimeError(
                "MDRSScorer has not been fitted. Call fit() first."
            )

        X = self._subsample(states)
        diff = X - self.mu

        # MDRS-03: vectorised squared Mahalanobis distance
        return np.einsum("ij,jk,ik->i", diff, self.precision, diff)

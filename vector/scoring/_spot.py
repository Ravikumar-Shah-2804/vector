"""Streaming Peaks Over Threshold (SPOT) for automatic threshold selection.

Adapted from the SPOT algorithm (Siffer et al., KDD 2017) as implemented
in TransNAS-TSAD. Stripped of matplotlib/pandas dependencies. Core Grimshaw
GPD estimation preserved.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize


class SPOT:
    """Streaming Peaks Over Threshold using GPD tail estimation.

    Parameters
    ----------
    q : float
        Risk parameter controlling the false-alarm rate. Lower q gives a
        higher (more conservative) threshold.
    """

    def __init__(self, q: float = 1e-4) -> None:
        self.q = q
        self.extreme_quantile: float | None = None
        self.init_data: np.ndarray | None = None
        self.data: np.ndarray | None = None
        self.init_threshold: float | None = None
        self.peaks: np.ndarray | None = None
        self.n: int = 0
        self.Nt: int = 0

    def fit(self, init_data: np.ndarray, data: np.ndarray) -> None:
        """Store calibration and stream data.

        Parameters
        ----------
        init_data : np.ndarray
            1D array of training scores for threshold calibration.
        data : np.ndarray
            1D array of validation scores to threshold.
        """
        self.init_data = np.sort(init_data)
        self.data = data
        self.n = len(init_data)

    def initialize(
        self,
        level: float = 0.98,
        min_extrema: bool = False,
        verbose: bool = False,
    ) -> None:
        """Calibrate initial threshold and fit GPD to exceedances.

        Parameters
        ----------
        level : float
            Quantile level for initial threshold on calibration data.
        min_extrema : bool
            If True, use lower tail instead of upper tail.
        verbose : bool
            Unused. Kept for interface compatibility.

        Raises
        ------
        ValueError
            If no peaks (exceedances) found above the initial threshold.
        """
        if self.init_data is None:
            raise RuntimeError("Call fit() before initialize().")

        n = len(self.init_data)
        idx = int(level * n)
        idx = min(idx, n - 1)
        self.init_threshold = self.init_data[idx]

        # Extract peaks (exceedances above threshold)
        if min_extrema:
            peaks = self.init_threshold - self.init_data[
                self.init_data < self.init_threshold
            ]
        else:
            peaks = (
                self.init_data[self.init_data > self.init_threshold]
                - self.init_threshold
            )

        if len(peaks) == 0:
            raise ValueError(
                f"No exceedances found above threshold {self.init_threshold:.6f} "
                f"at level {level:.4f}. Data may be degenerate."
            )

        self.peaks = peaks
        self.Nt = len(peaks)

        gamma, sigma = self._grimshaw(peaks, self.init_threshold)
        self.extreme_quantile = self._quantile(gamma, sigma)

    def run(self, dynamic: bool = False) -> dict:
        """Apply threshold to stream data.

        Parameters
        ----------
        dynamic : bool
            If True, update threshold as new data arrives.

        Returns
        -------
        dict
            Keys: 'thresholds' (list of float), 'alarms' (list of int).
        """
        if self.extreme_quantile is None:
            raise RuntimeError("Call initialize() before run().")

        if self.data is None:
            raise RuntimeError("No stream data provided.")

        thresholds = []
        alarms = []
        threshold = self.extreme_quantile

        for i, val in enumerate(self.data):
            if dynamic and val > self.init_threshold:
                self.peaks = np.append(self.peaks, val - self.init_threshold)
                self.Nt += 1
                self.n += 1
                gamma, sigma = self._grimshaw(self.peaks, self.init_threshold)
                threshold = self._quantile(gamma, sigma)

            if val > threshold:
                alarms.append(i)

            thresholds.append(threshold)

        return {"thresholds": thresholds, "alarms": alarms}

    def _grimshaw(
        self, peaks: np.ndarray, threshold: float
    ) -> tuple[float, float]:
        """Estimate GPD parameters via Grimshaw's MLE method.

        Parameters
        ----------
        peaks : np.ndarray
            Exceedances above threshold.
        threshold : float
            Initial threshold value.

        Returns
        -------
        tuple[float, float]
            (gamma, sigma) GPD parameters.
        """
        Nt = len(peaks)
        if Nt == 0:
            raise ValueError("Empty peaks array.")

        min_peak = peaks.min()
        max_peak = peaks.max()
        mean_peak = peaks.mean()

        if max_peak == min_peak:
            # Constant peaks, use exponential approximation
            return 0.0, mean_peak if mean_peak > 0 else 1e-6

        # Grimshaw: find sigma and gamma from the MLE equations
        # Using the profile likelihood approach
        def neg_log_lik(params: np.ndarray) -> float:
            gamma = params[0]
            sigma = params[1]
            if sigma <= 0:
                return 1e10
            if gamma == 0:
                return Nt * math.log(sigma) + np.sum(peaks) / sigma
            arg = 1.0 + gamma * peaks / sigma
            if np.any(arg <= 0):
                return 1e10
            return Nt * math.log(sigma) + (1.0 + 1.0 / gamma) * np.sum(
                np.log(arg)
            )

        # Multiple starting points for robustness
        best_result = None
        best_nll = float("inf")

        for g0 in [-0.1, 0.0, 0.1, 0.5]:
            for s0_mult in [0.5, 1.0, 2.0]:
                s0 = mean_peak * s0_mult
                if s0 <= 0:
                    s0 = 1e-4
                try:
                    result = minimize(
                        neg_log_lik,
                        x0=np.array([g0, s0]),
                        method="L-BFGS-B",
                        bounds=[(-1.0, 10.0), (1e-10, None)],
                    )
                    if result.success and result.fun < best_nll:
                        best_nll = result.fun
                        best_result = result
                except (ValueError, RuntimeWarning, FloatingPointError):
                    continue

        if best_result is None:
            # Fallback: assume exponential (gamma=0)
            return 0.0, mean_peak if mean_peak > 0 else 1e-6

        gamma = float(best_result.x[0])
        sigma = float(best_result.x[1])
        return gamma, sigma

    def _quantile(self, gamma: float, sigma: float) -> float:
        """Compute extreme quantile from GPD parameters.

        Parameters
        ----------
        gamma : float
            Shape parameter.
        sigma : float
            Scale parameter.

        Returns
        -------
        float
            Extreme quantile threshold.
        """
        if self.n == 0 or self.Nt == 0:
            return self.init_threshold or 0.0

        r = self.n * self.q / self.Nt

        if gamma != 0:
            return self.init_threshold + (sigma / gamma) * (
                math.pow(r, -gamma) - 1.0
            )
        else:
            return self.init_threshold - sigma * math.log(r)

    def _log_likelihood(
        self, Y: np.ndarray, gamma: float, sigma: float
    ) -> float:
        """GPD log-likelihood for peaks Y.

        Parameters
        ----------
        Y : np.ndarray
            Peak values.
        gamma : float
            Shape parameter.
        sigma : float
            Scale parameter.

        Returns
        -------
        float
            Log-likelihood value.
        """
        n = len(Y)
        if sigma <= 0:
            return -float("inf")
        if gamma == 0:
            return -n * math.log(sigma) - np.sum(Y) / sigma
        arg = 1.0 + gamma * Y / sigma
        if np.any(arg <= 0):
            return -float("inf")
        return -n * math.log(sigma) - (1.0 + 1.0 / gamma) * np.sum(
            np.log(arg)
        )

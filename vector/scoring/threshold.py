"""SPOT-based automatic threshold selection with retry and fallback."""

from __future__ import annotations

import numpy as np

from vector.scoring._spot import SPOT


class SPOTThreshold:
    """Automatic threshold selection using SPOT with retry and percentile fallback.

    Parameters
    ----------
    q : float
        SPOT risk parameter.
    level : float
        Initial quantile level for SPOT calibration.
    scaling_factor : float
        Multiplier applied to the final threshold.
    fallback_percentile : float
        Percentile used when SPOT fails after all retries.
    """

    def __init__(
        self,
        q: float = 1e-4,
        level: float = 0.98,
        scaling_factor: float = 1.0,
        fallback_percentile: float = 99.5,
    ) -> None:
        self.q = q
        self.level = level
        self.scaling_factor = scaling_factor
        self.fallback_percentile = fallback_percentile
        self.threshold: float | None = None
        self.used_fallback: bool = False

    def fit(
        self, train_scores: np.ndarray, val_scores: np.ndarray
    ) -> SPOTThreshold:
        """Fit threshold on train and validation scores.

        Uses SPOT with up to 50 retry attempts. Each retry decrements the
        level by 0.001. Falls back to percentile threshold only after all
        retries are exhausted or a non-Grimshaw exception occurs.

        Parameters
        ----------
        train_scores : np.ndarray
            1D array of training anomaly scores (for SPOT calibration).
        val_scores : np.ndarray
            1D array of validation anomaly scores (for threshold computation).

        Returns
        -------
        SPOTThreshold
            Self, for method chaining.
        """
        spot = SPOT(q=self.q)
        spot.fit(train_scores, val_scores)

        current_level = self.level
        max_retries = 50
        spot_succeeded = False

        for attempt in range(max_retries):
            try:
                spot.initialize(level=current_level)
                result = spot.run(dynamic=False)
                thresholds = result["thresholds"]
                if len(thresholds) > 0:
                    self.threshold = float(np.mean(thresholds))
                    spot_succeeded = True
                    break
            except (ValueError, RuntimeError, FloatingPointError):
                # Grimshaw/GPD related failure, retry with lower level
                current_level -= 0.001
                if current_level <= 0.01:
                    break
                continue
            except Exception:
                # Non-Grimshaw exception, stop retrying
                break

        if not spot_succeeded:
            self.threshold = float(
                np.percentile(val_scores, self.fallback_percentile)
            )
            self.used_fallback = True

        self.threshold *= self.scaling_factor
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """Produce binary anomaly predictions.

        Parameters
        ----------
        scores : np.ndarray
            1D array of anomaly scores.

        Returns
        -------
        np.ndarray
            Binary predictions of shape ``(n_samples,)`` with dtype int32.
        """
        if self.threshold is None:
            raise RuntimeError(
                "SPOTThreshold has not been fitted. Call fit() first."
            )
        return (scores > self.threshold).astype(np.int32)

    @classmethod
    def from_config(
        cls, dataset_name: str, config: dict | None = None
    ) -> SPOTThreshold:
        """Create SPOTThreshold with dataset-specific parameters from config.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset (e.g., 'NAB', 'SMD').
        config : dict or None
            Parsed datasets.yaml config. If None, loads from default path.

        Returns
        -------
        SPOTThreshold
            Instance with dataset-specific level and scaling_factor.
        """
        if config is None:
            from vector.data.config import load_config

            config = load_config()

        # Config may have a top-level 'datasets' key wrapping all entries
        datasets = config.get("datasets", config)
        ds_name = dataset_name.upper()
        ds_config = datasets.get(ds_name, datasets.get(dataset_name.lower(), {}))
        spot_config = ds_config.get("spot", {})

        level = spot_config.get("level", 0.98)
        scaling_factor = spot_config.get("scaling_factor", 1.0)

        return cls(level=level, scaling_factor=scaling_factor)

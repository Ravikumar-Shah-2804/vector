"""Echo State Network with sparse reservoir and leaky-integrator dynamics."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, ArpackNoConvergence


class EchoStateNetwork:
    """Leaky-integrator Echo State Network for time-series featurization.

    Constructs a sparse reservoir matrix scaled to a target spectral radius,
    then drives it with input windows to produce high-dimensional state
    representations. Deterministic when seeded.

    Parameters
    ----------
    n_input : int
        Dimensionality of input features.
    n_reservoir : int
        Number of reservoir neurons.
    spectral_radius : float
        Target spectral radius for the reservoir weight matrix.
    input_scaling : float
        Magnitude bound for input weight matrix entries.
    sparsity : float
        Density (fraction of non-zero entries) of the reservoir matrix.
    leak_rate : float
        Leaky-integrator coefficient in (0, 1].
    washout : int
        Number of initial timesteps to discard from the state output.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_input: int,
        n_reservoir: int = 500,
        spectral_radius: float = 0.9,
        input_scaling: float = 0.1,
        sparsity: float = 0.1,
        leak_rate: float = 0.3,
        washout: int = 50,
        seed: int = 42,
    ) -> None:
        self.n_reservoir = n_reservoir
        self.leak_rate = leak_rate
        self.washout = washout

        # Single RNG for all randomness (ESN-07: determinism).
        # Order matters: W_in first, then W.
        rng = np.random.default_rng(seed)

        # ESN-01: input weight matrix
        self.W_in: np.ndarray = rng.uniform(
            -input_scaling, input_scaling, (n_reservoir, n_input)
        )

        # ESN-02: sparse reservoir matrix
        W = sp.random(
            n_reservoir,
            n_reservoir,
            density=sparsity,
            random_state=rng,
            data_rvs=lambda s: rng.uniform(-1, 1, size=s),
            format="csr",
        )

        # ESN-03: scale to target spectral radius
        self.W = self._scale_spectral_radius(W, spectral_radius)

    @staticmethod
    def _scale_spectral_radius(
        W: sp.csr_matrix, target_rho: float, max_retries: int = 3
    ) -> sp.csr_matrix:
        """Scale *W* so its spectral radius equals *target_rho*.

        Uses ARPACK sparse eigensolver with increasing maxiter on retry.
        Falls back to dense eigvals if ARPACK fails to converge.

        Raises
        ------
        ValueError
            If the computed spectral radius is near zero (< 1e-10),
            making scaling numerically unstable.
        """
        sr: float | None = None

        # Try sparse eigensolver with increasing patience
        for attempt in range(max_retries):
            try:
                eigenvalues = eigs(
                    W.astype(np.float64),
                    k=1,
                    which="LM",
                    maxiter=5000 * (attempt + 1),
                    return_eigenvectors=False,
                )
                sr = float(np.max(np.abs(eigenvalues)))
                break
            except (ArpackNoConvergence, TypeError):
                # TypeError: k >= N-1 for tiny matrices; ARPACK can't handle it
                if attempt < max_retries - 1:
                    continue
                # Final retry failed -- fall back to dense
                sr = float(np.max(np.abs(np.linalg.eigvals(W.toarray()))))

        # ESN-08: guard against degenerate reservoir
        if sr is not None and sr < 1e-10:
            raise ValueError(
                f"Spectral radius near zero ({sr:.2e}). "
                "Increase reservoir size or sparsity."
            )

        return W * (target_rho / sr)

    def transform(self, windows: np.ndarray) -> np.ndarray:
        """Drive the reservoir with input windows and collect states.

        Parameters
        ----------
        windows : np.ndarray
            Input tensor of shape ``(N, window_size, D)`` where *N* is the
            number of windows and *D* is the input dimensionality.

        Returns
        -------
        np.ndarray
            Reservoir states of shape ``(N * window_size - washout, n_reservoir)``
            after discarding the washout transient (ESN-05, ESN-06).
        """
        N, window_size, _ = windows.shape
        T = N * window_size

        # Pre-allocate state collection matrix
        states = np.empty((T, self.n_reservoir))

        # ESN-04: leaky-integrator with continuous state across windows
        x = np.zeros(self.n_reservoir)
        t = 0
        for i in range(N):
            for j in range(window_size):
                u = windows[i, j]  # shape (D,)
                x = (1 - self.leak_rate) * x + self.leak_rate * np.tanh(
                    self.W_in @ u + self.W @ x
                )
                states[t] = x
                t += 1

        # ESN-05: discard washout transient
        return states[self.washout :]

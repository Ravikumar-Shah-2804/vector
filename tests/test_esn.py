"""Comprehensive test suite for EchoStateNetwork."""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

from vector.esn import EchoStateNetwork


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_esn():
    """ESN with moderate size for general tests."""
    return EchoStateNetwork(n_input=5, n_reservoir=100, seed=42)


@pytest.fixture
def rng():
    """Shared RNG for deterministic test data."""
    return np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Spectral radius
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rho", [0.5, 0.9, 1.2, 1.5])
def test_spectral_radius_within_tolerance(rho):
    """Constructed W must have spectral radius within 1% of target."""
    esn = EchoStateNetwork(
        n_input=3, n_reservoir=200, spectral_radius=rho, seed=7
    )
    eigenvalues = eigs(esn.W.astype(np.float64), k=1, which="LM",
                       return_eigenvectors=False)
    actual_sr = float(np.max(np.abs(eigenvalues)))
    assert abs(actual_sr - rho) / rho < 0.01, (
        f"Spectral radius {actual_sr:.6f} not within 1% of target {rho}"
    )


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n_input, n_res, washout, n_windows, win_size",
    [
        (5, 100, 50, 20, 30),
        (1, 50, 10, 10, 20),
        (10, 200, 0, 5, 15),
    ],
)
def test_output_shape(n_input, n_res, washout, n_windows, win_size):
    """Transform output shape must be (N*window_size - washout, n_reservoir)."""
    esn = EchoStateNetwork(
        n_input=n_input, n_reservoir=n_res, washout=washout, seed=0
    )
    rng = np.random.default_rng(0)
    windows = rng.standard_normal((n_windows, win_size, n_input))
    X = esn.transform(windows)
    expected = (n_windows * win_size - washout, n_res)
    assert X.shape == expected, f"Expected {expected}, got {X.shape}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_determinism_same_seed():
    """Two ESNs with the same seed must produce identical states."""
    esn1 = EchoStateNetwork(n_input=3, n_reservoir=50, seed=42)
    esn2 = EchoStateNetwork(n_input=3, n_reservoir=50, seed=42)
    rng = np.random.default_rng(0)
    windows = rng.standard_normal((4, 10, 3))
    X1 = esn1.transform(windows)
    X2 = esn2.transform(windows)
    assert np.allclose(X1, X2), "Same seed must yield identical states"


def test_different_seeds():
    """Different seeds must produce different reservoir states."""
    esn1 = EchoStateNetwork(n_input=3, n_reservoir=50, washout=0, seed=42)
    esn2 = EchoStateNetwork(n_input=3, n_reservoir=50, washout=0, seed=99)
    rng = np.random.default_rng(0)
    windows = rng.standard_normal((4, 10, 3))
    X1 = esn1.transform(windows)
    X2 = esn2.transform(windows)
    assert not np.allclose(X1, X2), "Different seeds must differ"


# ---------------------------------------------------------------------------
# Different inputs -> different states
# ---------------------------------------------------------------------------

def test_different_inputs_different_states():
    """Different input signals must produce different reservoir states."""
    esn = EchoStateNetwork(n_input=3, n_reservoir=50, washout=0, seed=42)
    zeros = np.zeros((4, 10, 3))
    rng = np.random.default_rng(0)
    rand = rng.standard_normal((4, 10, 3))
    X_zero = esn.transform(zeros)
    # Re-create to reset internal state (transform is stateless per call)
    esn2 = EchoStateNetwork(n_input=3, n_reservoir=50, washout=0, seed=42)
    X_rand = esn2.transform(rand)
    assert not np.allclose(X_zero, X_rand), "Different inputs must differ"


# ---------------------------------------------------------------------------
# Near-zero spectral radius
# ---------------------------------------------------------------------------

def test_near_zero_spectral_radius():
    """Near-zero spectral radius must raise ValueError."""
    # Build a diagonal matrix with near-zero entries
    W = sp.csr_matrix(np.diag([1e-15] * 20))
    with pytest.raises(ValueError, match="near zero"):
        EchoStateNetwork._scale_spectral_radius(W, target_rho=0.9)


# ---------------------------------------------------------------------------
# Edge-case rho values
# ---------------------------------------------------------------------------

def test_edge_case_rho_small():
    """rho=0.1 should work and produce small-magnitude states."""
    esn = EchoStateNetwork(
        n_input=3, n_reservoir=100, spectral_radius=0.1, washout=0, seed=0
    )
    rng = np.random.default_rng(0)
    windows = rng.standard_normal((4, 10, 3))
    X = esn.transform(windows)
    assert np.all(np.isfinite(X))
    assert np.max(np.abs(X)) < 0.5, "Small rho should keep states small"


def test_edge_case_rho_large():
    """rho=1.5 should produce valid finite output."""
    esn = EchoStateNetwork(
        n_input=3, n_reservoir=100, spectral_radius=1.5, washout=0, seed=0
    )
    rng = np.random.default_rng(0)
    windows = rng.standard_normal((4, 10, 3))
    X = esn.transform(windows)
    assert np.all(np.isfinite(X)), "rho=1.5 must not overflow"


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------

def test_reservoir_is_sparse(default_esn):
    """Reservoir weight matrix must be a sparse matrix."""
    assert sp.issparse(default_esn.W), "W must be scipy sparse"


def test_input_weights(rng):
    """W_in shape and values must respect input_scaling."""
    n_input, n_res, scaling = 5, 80, 0.5
    esn = EchoStateNetwork(
        n_input=n_input, n_reservoir=n_res,
        input_scaling=scaling, seed=0
    )
    assert esn.W_in.shape == (n_res, n_input)
    assert np.all(np.abs(esn.W_in) <= scaling), (
        f"W_in entries must lie in [-{scaling}, {scaling}]"
    )

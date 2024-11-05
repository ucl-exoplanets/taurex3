"""Test main contribution function."""
import numpy as np
import pytest

import taurex.contributions.contribution as contrib


def test_contribute_numpy():
    """Tests the numpy version of the contribution function."""
    sigma = np.random.rand(100, 200)
    density = np.random.rand(100)
    path = np.random.rand(100)
    tau = np.zeros((100, 200))
    contrib.contribute_tau_numpy(0, 100, 0, sigma, density, path, 100, 200, 0, tau)

    assert not np.all(tau == 0.0)


def test_contribute_numba():
    """Tests the numba version of the contribution function."""
    numba = pytest.importorskip("numba")  # noqa: F841
    sigma = np.random.rand(100, 200)
    density = np.random.rand(100)
    path = np.random.rand(100)
    tau = np.zeros((100, 200))
    contrib.contribute_tau_numba(0, 100, 0, sigma, density, path, 100, 200, 0, tau)

    assert not np.all(tau == 0.0)


def test_contribute_consistent():
    """Tests the consistency of the numpy and numba contribution functions."""
    numba = pytest.importorskip("numba")  # noqa: F841
    sigma = np.random.rand(100, 200)
    density = np.random.rand(100)

    tau1 = np.zeros((100, 200))
    tau2 = np.zeros((100, 200))

    for x in range(100):
        path = np.random.rand(100 - x)
        contrib.contribute_tau_numpy(
            0, 100 - x, x, sigma, density, path, 100, 200, x, tau1
        )
        contrib.contribute_tau_numba(
            0, 100 - x, x, sigma, density, path, 100, 200, x, tau2
        )

    np.testing.assert_array_equal(tau1, tau2)

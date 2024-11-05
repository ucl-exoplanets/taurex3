"""Test linear interpolation."""

import pytest


@pytest.mark.bench
def test_linear_interp_numpy(benchmark):
    """Test linear interpolation using numpy."""
    import numpy as np

    from taurex.util.math import interp_lin_numpy

    # intepr_bilin_old(x11, x12, x21, x22, T, Tmin, Tmax, P, Pmin, Pmax)

    x11 = np.random.rand(10000)
    x12 = np.random.rand(10000)

    benchmark(interp_lin_numpy, x11, x12, 0.5, 0.1, 1.0)


@pytest.mark.bench
def test_linear_interp_numba(benchmark):
    """Test linear interpolation using numba."""
    import numpy as np

    _ = pytest.importorskip("numba")
    from taurex.util.math import interp_lin_numba, interp_lin_numpy

    # intepr_bilin_old(x11, x12, x21, x22, T, Tmin, Tmax, P, Pmin, Pmax)

    x11 = np.random.rand(10000)
    x12 = np.random.rand(10000)

    benchmark(interp_lin_numba, x11, x12, 0.5, 0.1, 1.0)

    np.testing.assert_array_almost_equal(
        interp_lin_numba(x11, x12, 0.5, 0.1, 1.0),
        interp_lin_numpy(x11, x12, 0.5, 0.1, 1.0),
    )


@pytest.mark.bench
def test_exp_interp_numpy(benchmark):
    """Test exponential interpolation using numpy."""
    import numpy as np

    from taurex.util.math import interp_exp_numpy

    # intepr_bilin_old(x11, x12, x21, x22, T, Tmin, Tmax, P, Pmin, Pmax)

    x11 = np.random.rand(10000)
    x12 = np.random.rand(10000)

    benchmark(interp_exp_numpy, x11, x12, 0.5, 0.1, 1.0)


@pytest.mark.bench
def test_exp_interp_numba(benchmark):
    """Test exponential interpolation using numba."""
    import numpy as np

    _ = pytest.importorskip("numba")
    from taurex.util.math import interp_exp_numba

    # intepr_bilin_old(x11, x12, x21, x22, T, Tmin, Tmax, P, Pmin, Pmax)

    x11 = np.random.rand(10000)
    x12 = np.random.rand(10000)

    benchmark(interp_exp_numba, x11, x12, 0.5, 0.1, 1.0)

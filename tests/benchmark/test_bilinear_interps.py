"""Test bilinear interpolation functions."""

import pytest

ARRAY_SIZE = 100000


@pytest.mark.bench
def test_bilin_interp_numpy(benchmark):
    """Test bilinear interpolation using numpy."""
    import numpy as np

    from taurex.util.math import intepr_bilin_old

    # intepr_bilin_old(x11, x12, x21, x22, T, Tmin, Tmax, P, Pmin, Pmax)

    x11 = np.random.rand(ARRAY_SIZE)
    x12 = np.random.rand(ARRAY_SIZE)
    x21 = np.random.rand(ARRAY_SIZE)
    x22 = np.random.rand(ARRAY_SIZE)

    benchmark(intepr_bilin_old, x11, x12, x21, x22, 0.5, 0.1, 1.0, 0.5, 0.1, 1.0)


@pytest.mark.bench
def test_bilin_interp_numba(benchmark):
    """Test bilinear interpolation using numba."""
    import numpy as np

    _ = pytest.importorskip("numba")
    from taurex.util.math import intepr_bilin_numba

    # intepr_bilin_old(x11, x12, x21, x22, T, Tmin, Tmax, P, Pmin, Pmax)

    x11 = np.random.rand(ARRAY_SIZE)
    x12 = np.random.rand(ARRAY_SIZE)
    x21 = np.random.rand(ARRAY_SIZE)
    x22 = np.random.rand(ARRAY_SIZE)

    benchmark(intepr_bilin_numba, x11, x12, x21, x22, 0.5, 0.1, 1.0, 0.5, 0.1, 1.0)


@pytest.mark.bench
def test_bilin_interp_numba_ii(benchmark):
    """Test bilinear interpolation using numba."""
    import numpy as np

    _ = pytest.importorskip("numba")
    from taurex.util.math import intepr_bilin_numba, intepr_bilin_numba_II

    # intepr_bilin_old(x11, x12, x21, x22, T, Tmin, Tmax, P, Pmin, Pmax)

    x11 = np.random.rand(ARRAY_SIZE)
    x12 = np.random.rand(ARRAY_SIZE)
    x21 = np.random.rand(ARRAY_SIZE)
    x22 = np.random.rand(ARRAY_SIZE)

    benchmark(intepr_bilin_numba_II, x11, x12, x21, x22, 0.5, 0.1, 1.0, 0.5, 0.1, 1.0)

    np.testing.assert_array_almost_equal(
        intepr_bilin_numba_II(x11, x12, x21, x22, 1.2, 0.1, 2.0, 3.8, 0.1, 5.0),
        intepr_bilin_numba(x11, x12, x21, x22, 1.2, 0.1, 2.0, 3.8, 0.1, 5.0),
    )


@pytest.mark.bench
def test_bilin_interp_double(benchmark):
    """Test bilinear interpolation by running linear twice."""
    import numpy as np

    from taurex.util.math import intepr_bilin_double, intepr_bilin_old

    # intepr_bilin_old(x11, x12, x21, x22, T, Tmin, Tmax, P, Pmin, Pmax)

    x11 = np.random.rand(ARRAY_SIZE)
    x12 = np.random.rand(ARRAY_SIZE)
    x21 = np.random.rand(ARRAY_SIZE)
    x22 = np.random.rand(ARRAY_SIZE)

    benchmark(intepr_bilin_double, x11, x12, x21, x22, 0.5, 0.1, 1.0, 0.5, 0.1, 1.0)

    np.testing.assert_array_almost_equal(
        intepr_bilin_double(x11, x12, x21, x22, 0.2, 0.1, 1.0, 0.8, 0.1, 1.0),
        intepr_bilin_old(x11, x12, x21, x22, 0.2, 0.1, 1.0, 0.8, 0.1, 1.0),
    )


@pytest.mark.bench
def test_bilin_interp_numexpr(benchmark):
    """Test bilinear interpolation using numexpr."""
    import numpy as np

    from taurex.util.math import intepr_bilin_numexpr

    _ = pytest.importorskip("numexpr")

    x11 = np.random.rand(ARRAY_SIZE)
    x12 = np.random.rand(ARRAY_SIZE)
    x21 = np.random.rand(ARRAY_SIZE)
    x22 = np.random.rand(ARRAY_SIZE)
    # Cache first result
    intepr_bilin_numexpr(x11, x12, x21, x22, 0.5, 0.1, 1.0, 0.5, 0.1, 1.0)
    benchmark(intepr_bilin_numexpr, x11, x12, x21, x22, 0.5, 0.1, 1.0, 0.5, 0.1, 1.0)

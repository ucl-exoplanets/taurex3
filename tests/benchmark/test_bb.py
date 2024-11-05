"""Test blackbody emission functions."""

import numpy as np
import pytest

ARRAY_SIZE = 10000
NLAYERS = 100
N_MU = 10


@pytest.fixture
def wngrid():
    """Wavenumber grid."""
    yield np.linspace(300, 30000, ARRAY_SIZE)


@pytest.fixture
def temperature():
    """Temperature fixture."""
    yield np.linspace(1500, 1000, NLAYERS)


@pytest.mark.bench
@pytest.mark.slow
def test_bb_numpy(benchmark, wngrid):
    """Test blackbody emission using numpy."""
    from taurex.util.emission import black_body_numpy

    benchmark(black_body_numpy, wngrid, 1000)


@pytest.mark.bench
def test_bb_numexpr(benchmark, wngrid):
    """Test blackbody emission using numexpr."""
    from taurex.util.emission import black_body_numexpr

    _ = pytest.importorskip("numexpr")
    black_body_numexpr(wngrid, 1000)
    benchmark(black_body_numexpr, wngrid, 1000)


@pytest.mark.bench
def test_bb_numba(benchmark, wngrid):
    """Test blackbody emission using numba."""
    from taurex.util.emission import black_body_numba

    _ = pytest.importorskip("numba")
    black_body_numba(wngrid, 1000)
    benchmark(black_body_numba, wngrid, 1000)


@pytest.mark.bench
def test_bb_numba_ii(benchmark, wngrid):
    """Test blackbody emission using numba."""
    from taurex.util.emission import black_body_numba_II

    _ = pytest.importorskip("numba")

    benchmark(black_body_numba_II, wngrid, 1000)

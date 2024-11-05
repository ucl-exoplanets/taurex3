"""Test optical depth computation."""

import numpy as np
import pytest

from taurex.contributions import contribute_tau

NLAYERS = 100
WNGRID_SIZE = 100000


@pytest.fixture
def setup():
    """Setup for tests."""
    sigma = np.random.rand(NLAYERS, WNGRID_SIZE)
    density = np.random.rand(NLAYERS)
    path = np.random.rand(NLAYERS)
    yield sigma, density, path


def contribute_tau_numpy(
    start_k, end_k, density_offset, sigma, density, path, nlayers, ngrid, layer, tau
):
    """Contribute tau using numpy."""
    _path = path[start_k:end_k, None]
    _density = density[start_k + density_offset : end_k + density_offset, None]
    _sigma = sigma[start_k + layer : end_k + layer]
    # for mol in range(nmols):
    tau[layer] += np.sum(_path * _density * _sigma, axis=0)


def contribute_tau_numexpr(
    start_k, end_k, density_offset, sigma, density, path, nlayers, ngrid, layer, tau
):
    """Contribute tau using numexpr."""
    ne = pytest.importorskip("numexpr")

    _path = path[start_k:end_k, None]  # noqa: F841
    _density = density[  # noqa: F841
        start_k + density_offset : end_k + density_offset, None
    ]
    _sigma = sigma[start_k + layer : end_k + layer]  # noqa: F841
    # for mol in range(nmols):
    tau[layer] += ne.evaluate("sum(_path*_density*_sigma, axis=0)")


@pytest.mark.bench
def test_contribute_tau_numba(benchmark, setup):
    """Test contribute_tau using numba."""
    _ = pytest.importorskip("numba")
    sigma, density, path = setup
    # # start_k, end_k, density_offset, sigma, density, path, nlayers,
    #                ngrid, layer, tau
    tau = np.zeros(shape=(NLAYERS, WNGRID_SIZE))
    benchmark(
        contribute_tau,
        0,
        NLAYERS,
        0,
        sigma,
        density,
        path,
        NLAYERS,
        WNGRID_SIZE,
        0,
        tau,
    )


@pytest.mark.bench
def test_contribute_tau_numba_ii(benchmark, setup):
    """Test contribute_tau using numba (alt)."""
    numba = pytest.importorskip("numba")

    @numba.vectorize([numba.float64(numba.float64, numba.float64)])
    def _integrate(x1, x2):
        return x1 * x2

    @numba.jit(nopython=True, nogil=True, parallel=False)
    def contribute_tau_ii(
        start_k, end_k, density_offset, sigma, density, path, nlayers, ngrid, layer, tau
    ):
        _d = density[density_offset:]
        for k in range(start_k, end_k):
            _density = _d[k] * path[k]
            sig = sigma[k + layer]
            # for mol in range(nmols):
            out = _integrate(sig, _density)
            # for wn in range(ngrid):
            tau[layer] += out

    sigma, density, path = setup
    # # start_k, end_k, density_offset, sigma, density, path, nlayers,
    #                ngrid, layer, tau
    tau = np.zeros(shape=(NLAYERS, WNGRID_SIZE))
    benchmark(
        contribute_tau_ii,
        0,
        NLAYERS,
        0,
        sigma,
        density,
        path,
        NLAYERS,
        WNGRID_SIZE,
        0,
        tau,
    )


# def test_contribute_tau_cython(benchmark, setup):
#     from taurex.external.contrib import contrib_tau_cython
#     sigma, density, path = setup
#     # # start_k, end_k, density_offset, sigma, density, path, nlayers,
#     #                ngrid, layer, tau
#     tau = np.zeros(shape=(NLAYERS, WNGRID_SIZE))
#     benchmark(contrib_tau_cython, 0, NLAYERS, 0, sigma, density, path, NLAYERS,
#               WNGRID_SIZE, 0, tau)
#     tau_1  = np.zeros(shape=(1, WNGRID_SIZE))
#     tau_2  = np.zeros(shape=(1, WNGRID_SIZE))
#     contrib_tau_cython(0, NLAYERS, 0, sigma, density, path, NLAYERS,
#               WNGRID_SIZE, 0, tau_1)

#     contribute_tau(0, NLAYERS, 0, sigma, density, path, NLAYERS,
#               WNGRID_SIZE, 0, tau_2)


#     np.testing.assert_array_almost_equal(tau_1,tau_2)
@pytest.mark.bench
def test_contribute_tau_numpy(benchmark, setup):
    """Test contribute_tau using numpy."""
    sigma, density, path = setup
    # # start_k, end_k, density_offset, sigma, density, path, nlayers,
    #                ngrid, layer, tau
    tau = np.zeros(shape=(NLAYERS, WNGRID_SIZE))
    benchmark(
        contribute_tau_numpy,
        0,
        NLAYERS,
        0,
        sigma,
        density,
        path,
        NLAYERS,
        WNGRID_SIZE,
        0,
        tau,
    )


@pytest.mark.bench
def test_contribute_tau_numexpr(benchmark, setup):
    """Test contribute_tau using numexpr."""
    sigma, density, path = setup
    # # start_k, end_k, density_offset, sigma, density, path, nlayers,
    #                ngrid, layer, tau
    tau = np.zeros(shape=(NLAYERS, WNGRID_SIZE))
    benchmark(
        contribute_tau_numexpr,
        0,
        NLAYERS,
        0,
        sigma,
        density,
        path,
        NLAYERS,
        WNGRID_SIZE,
        0,
        tau,
    )

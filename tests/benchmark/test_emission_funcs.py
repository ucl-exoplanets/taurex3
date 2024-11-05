"""Test emission functions."""

import numpy as np
import pytest

ARRAY_SIZE = 30000
NLAYERS = 500
N_MU = 4


@pytest.fixture
def wngrid():
    """Wavenumber grid."""
    yield np.linspace(300, 30000, ARRAY_SIZE)


@pytest.fixture
def temperature():
    """Temperature."""
    yield np.linspace(1500, 1000, NLAYERS)


@pytest.mark.bench
def test_integrate_emission(benchmark, wngrid, temperature):
    """Test integrate_emission."""
    from taurex.util.emission import black_body, integrate_emission_layer

    dtau = np.ones(shape=(1, wngrid.shape[0]))
    ltau = np.ones(shape=(1, wngrid.shape[0]))
    mu = np.ones(N_MU)

    def integrate_nlayers():
        for n in range(NLAYERS):
            integrate_emission_layer(dtau, ltau, mu, black_body(wngrid, temperature[n]))

    benchmark(integrate_nlayers)

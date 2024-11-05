"""Test emission utilities."""

import astropy.units as u
import hypothesis
import numpy as np
import pytest
from astropy.modeling.models import BlackBody
from hypothesis.strategies import floats

from taurex.util.emission import black_body, black_body_numpy


@hypothesis.given(floats(300, 10000))
@hypothesis.settings(deadline=400)
def test_blackbody(temperature):
    """Test default blackbody function."""
    wngrid = np.linspace(200, 30000, 200)

    bb = black_body(wngrid, temperature) / np.pi

    bb_as = BlackBody(temperature=temperature * u.K)
    expect_flux = bb_as(wngrid * u.k).to(
        u.W / u.m**2 / u.micron / u.sr, equivalencies=u.spectral_density(wngrid * u.k)
    )

    assert bb == pytest.approx(expect_flux.value, rel=1e-3)


@hypothesis.given(floats(300, 10000))
def test_blackbody_numpy(temperature):
    """Test numpy blackbody function."""
    wngrid = np.linspace(200, 30000, 20)

    bb = black_body_numpy(wngrid, temperature) / np.pi

    assert bb.shape == (20,)
    bb_as = BlackBody(temperature=temperature * u.K)
    expect_flux = bb_as(wngrid * u.k).to(
        u.W / u.m**2 / u.micron / u.sr, equivalencies=u.spectral_density(wngrid * u.k)
    )

    assert bb == pytest.approx(expect_flux.value, rel=1e-3)


def test_blackbody_numpy_multitemp():
    """Test numpy blackbody function."""
    wngrid = np.linspace(200, 30000, 20)
    temperature = np.linspace(1500, 1000, 5)
    bb = black_body_numpy(wngrid, temperature) / np.pi
    assert bb.shape == (5, 20)

    bb = black_body_numpy(wngrid[0], temperature) / np.pi

    assert bb.shape == (5,)

    bb = black_body_numpy(wngrid, temperature[0]) / np.pi

    assert bb.shape == (20,)

    bb = black_body_numpy(wngrid[0], temperature[0]) / np.pi

    assert isinstance(bb, float)

    temperature = np.linspace(1500, 1000, 10).reshape(5, 2)
    wngrid = np.linspace(200, 30000, 20).reshape(2, 10)

    bb = black_body_numpy(wngrid, temperature) / np.pi

    assert bb.shape == (5, 2, 2, 10)

"""Tests for Phoenix4AllStar."""
from unittest.mock import patch

import numpy as np
import pytest
from astropy import units as u

from taurex.data.stellar.phoenix4all import Phoenix4AllStar

RSOL = 6.957e8  # Solar radius in metres


def test_phoenix4all_input_keywords():
    """Phoenix4AllStar registers under 'phoenix4all' keyword."""
    assert "phoenix4all" in Phoenix4AllStar.input_keywords()


def test_phoenix4all_creation():
    """Can instantiate Phoenix4AllStar with default parameters."""
    star = Phoenix4AllStar(temperature=5778, radius=1.0, metallicity=0.0)
    assert star.temperature == 5778
    # Star base class converts radius to metres internally
    assert star.radius == pytest.approx(RSOL)
    assert star.metallicity == 0.0
    assert star.alpha == 0.0


def test_phoenix4all_logg_computed():
    """Surface gravity is computed from mass and radius when not given."""
    star = Phoenix4AllStar(temperature=5000, radius=1.0, mass=1.0)
    # logg ~ 4.44 for Sun-like star
    assert star.logg == pytest.approx(4.44, abs=0.1)


def test_phoenix4all_logg_custom():
    """Surface gravity can be explicitly specified."""
    star = Phoenix4AllStar(temperature=5000, radius=1.0, logg=4.5)
    assert star.logg == 4.5


@patch("taurex.data.stellar.phoenix4all.get_spectrum")
def test_phoenix4all_initialize(mock_get_spectrum):
    """initialize() calls get_spectrum and produces SED on wngrid."""
    nw = 10000
    wlgrid = np.linspace(0.3, 30.0, nw) * u.um
    flux = np.ones(nw) * 1e-9 * u.W / (u.m**2 * u.um)
    mock_get_spectrum.return_value = (wlgrid, flux)

    star = Phoenix4AllStar(temperature=5778, radius=1.0, metallicity=0.0)
    wngrid = np.linspace(300, 30000, 5000)
    star.initialize(wngrid)

    assert star.sed is not None
    assert star.sed.shape == (5000,)
    assert np.all(np.isfinite(star.sed))


@patch("taurex.data.stellar.phoenix4all.get_spectrum")
def test_phoenix4all_source_keyword(mock_get_spectrum):
    """Pass source keyword correctly."""
    nw = 1000
    wlgrid = np.linspace(0.3, 30.0, nw) * u.um
    flux = np.ones(nw) * 1e-9 * u.W / (u.m**2 * u.um)
    mock_get_spectrum.return_value = (wlgrid, flux)

    star = Phoenix4AllStar(temperature=5000, radius=1.0, metallicity=0.0,
                           source="synphot")
    wngrid = np.linspace(300, 30000, 1000)
    star.initialize(wngrid)

    assert star.sed is not None

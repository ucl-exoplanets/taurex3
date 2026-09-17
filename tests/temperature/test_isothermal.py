"""Test isothermal temperature profile."""

import numpy as np
import pytest
from astropy import units as u
from hypothesis import given
from hypothesis.strategies import floats
from hypothesis.strategies import integers

from taurex.temperature import Isothermal


@given(
    temperature=floats(allow_nan=False),
    another_temperature=floats(allow_nan=False),
    nlayers=integers(1, 50),
)
def test_isothermal(temperature, another_temperature, nlayers):
    """Test isothermal."""
    iso = Isothermal(T=temperature)

    iso.initialize_profile(nlayers=nlayers)

    params = iso.fitting_parameters()

    assert iso.isoTemperature == temperature
    assert np.all(iso.profile == temperature)
    assert params["T"][2]() == temperature
    assert iso.profile.shape[0] == nlayers

    params["T"][3](another_temperature)

    assert iso.isoTemperature == another_temperature
    assert np.all(iso.profile == another_temperature)
    assert params["T"][2]() == another_temperature


def test_isothermal_accepts_quantity():
    """Quantity input is normalized to Kelvin in the generated profile."""
    iso = Isothermal(T=1.0 * u.kK)

    iso.initialize_profile(nlayers=3)

    assert iso.isoTemperature == pytest.approx(1000.0)
    np.testing.assert_allclose(iso.profile, [1000.0, 1000.0, 1000.0])


def test_isothermal_accepts_celsius_quantity():
    """Offset temperature units are converted using temperature equivalencies."""
    iso = Isothermal(T=u.Quantity(20.0, u.deg_C))

    assert iso.isoTemperature == pytest.approx(293.15)


def test_isothermal_setter_accepts_quantity():
    """The fittable temperature setter also accepts Quantity input."""
    iso = Isothermal()

    iso.isoTemperature = 0.5 * u.kK

    assert iso.isoTemperature == pytest.approx(500.0)


def test_isothermal_rejects_incompatible_quantity():
    """The temperature boundary rejects incompatible dimensions."""
    with pytest.raises(u.UnitConversionError):
        Isothermal(T=1.0 * u.m)

"""Tests for the Rodgers 2000 temperature profile."""

import astropy.units as u
import numpy as np
import pytest

from taurex.temperature import Rodgers2000
from taurex.types import get_float_dtype


def test_rodgers_accepts_temperature_quantity():
    """Layer temperatures are normalized to Kelvin."""
    profile = Rodgers2000(u.Quantity([0.0, 100.0], u.deg_C))

    np.testing.assert_allclose(profile.temperature_layers, [273.15, 373.15])
    assert profile.temperature_layers.dtype == get_float_dtype()


def test_rodgers_fitting_setter_accepts_quantity():
    """Generated layer setters normalize Quantity input."""
    profile = Rodgers2000([300.0, 400.0])
    parameters = profile.fitting_parameters()

    parameters["T_1"][3](0.5 * u.kK)

    assert parameters["T_1"][2]() == pytest.approx(500.0)


def test_rodgers_rejects_incompatible_quantity():
    """Layer temperatures reject incompatible physical dimensions."""
    with pytest.raises(u.UnitConversionError):
        Rodgers2000(np.array([1.0, 2.0]) * u.m)

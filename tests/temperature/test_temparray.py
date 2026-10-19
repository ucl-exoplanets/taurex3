"""Tests for array-based temperature profiles."""

import astropy.units as u
import numpy as np
import pytest

from taurex.temperature import TemperatureArray
from taurex.types import get_float_dtype


def test_temperature_array_preserves_unitless_kelvin_values():
    """Plain arrays retain their legacy interpretation in Kelvin."""
    profile = TemperatureArray(tp_array=[300, 400, 500])
    profile.initialize_profile(nlayers=3)

    np.testing.assert_allclose(profile.profile, [300.0, 400.0, 500.0])
    assert profile.profile.dtype == get_float_dtype()


def test_temperature_array_accepts_quantity():
    """Temperature arrays are normalized to Kelvin."""
    temperatures = u.Quantity([0.0, 100.0], u.deg_C)
    profile = TemperatureArray(tp_array=temperatures)
    profile.initialize_profile(nlayers=2)

    np.testing.assert_allclose(profile.profile, [273.15, 373.15])


def test_temperature_array_accepts_pressure_quantities():
    """Interpolation pressure points are normalized to Pa."""
    profile = TemperatureArray(
        tp_array=np.array([300.0, 500.0]) * u.K,
        p_points=np.array([1.0, 0.1]) * u.bar,
    )
    profile.initialize_profile(
        nlayers=2,
        pressure_profile=np.array([1e5, 1e4]),
    )

    np.testing.assert_allclose(profile.profile, [300.0, 500.0])


def test_temperature_array_reverses_normalized_values():
    """Reversal is applied after unit normalization."""
    profile = TemperatureArray(
        tp_array=np.array([300.0, 500.0]) * u.K,
        reverse=True,
    )
    profile.initialize_profile(nlayers=2)

    np.testing.assert_allclose(profile.profile, [500.0, 300.0])


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("tp_array", np.array([1.0, 2.0]) * u.m),
        ("p_points", np.array([1.0, 2.0]) * u.s),
    ],
)
def test_temperature_array_rejects_incompatible_quantities(argument, value):
    """Array boundaries reject incompatible physical dimensions."""
    kwargs = {"tp_array": [300.0, 400.0], argument: value}

    with pytest.raises(u.UnitConversionError):
        TemperatureArray(**kwargs)

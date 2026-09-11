"""Test pressure profile."""

import numpy as np
import pytest
from astropy import units as u
from hypothesis import example
from hypothesis import given
from hypothesis.strategies import floats
from hypothesis.strategies import integers

from taurex.data.profiles.pressure import LogPressureProfile
from taurex.pressure import ArrayPressureProfile
from taurex.pressure import PressureProfile
from taurex.pressure import SimplePressureProfile


@given(nlayers=integers())
def test_base_pressure(nlayers):
    """Test base pressure profile."""
    if nlayers <= 0:
        with pytest.raises(ValueError):
            p = PressureProfile("test", nlayers)
    else:
        p = PressureProfile("test", nlayers)

        assert p.nLayers == nlayers
        assert p.nLevels == nlayers + 1


@given(
    min_pressure=floats(1e4, 1e6),
    max_pressure=floats(1e-5, 1e5),
    nlayers=integers(-100, 400),
)
@example(min_pressure=10000.0, max_pressure=10000.000000000011, nlayers=1)
def test_simple_pressure(min_pressure, max_pressure, nlayers):
    """Test simple pressure profile."""
    if min_pressure >= max_pressure or nlayers <= 0:
        with pytest.raises(ValueError):
            sp = SimplePressureProfile(
                nlayers=nlayers,
                atm_min_pressure=min_pressure,
                atm_max_pressure=max_pressure,
            )
    else:
        sp = SimplePressureProfile(
            nlayers=nlayers,
            atm_min_pressure=min_pressure,
            atm_max_pressure=max_pressure,
        )

        assert sp.profile is None

        sp.compute_pressure_profile()

        pressure_profile = sp.profile

        pressure_profile_levels = sp.pressure_profile_levels
        assert pressure_profile_levels.argmax() == 0  # Ensure maximum is the first
        if nlayers > 1:
            assert pressure_profile_levels.argmin() == nlayers  # ensure minimum is last

        # When big floating point then max does not give exactly
        # the same as max pressure
        assert pressure_profile_levels.max() == pytest.approx(max_pressure, rel=1e-10)
        assert pressure_profile_levels.min() == pytest.approx(min_pressure, rel=1e-10)

        # Test to ensure they are bounded correctly
        assert np.all(pressure_profile_levels[1:-1] < max_pressure)
        assert np.all(pressure_profile_levels[1:-1] > min_pressure)

        # Test to ensure it is descreasing c
        assert np.all(np.diff(pressure_profile_levels) < 0)

        # Ensure pressure profile is always between levels

        assert np.all(pressure_profile_levels[:-1] > pressure_profile)
        assert np.all(pressure_profile_levels[1:] < pressure_profile)


def test_log_pressure_accepts_quantity_bounds():
    """Pressure bounds are normalized to Pa before building the profile."""
    profile = LogPressureProfile(
        nlayers=10,
        atm_min_pressure=1.0 * u.mbar,
        atm_max_pressure=2.0 * u.bar,
    )

    profile.compute_pressure_profile()

    assert profile.minAtmospherePressure == pytest.approx(100.0)
    assert profile.maxAtmospherePressure == pytest.approx(2e5)
    assert profile.pressure_profile_levels.min() == pytest.approx(100.0)
    assert profile.pressure_profile_levels.max() == pytest.approx(2e5)


def test_log_pressure_preserves_unitless_pa_bounds():
    """Plain pressure bounds retain their legacy interpretation in Pa."""
    profile = LogPressureProfile(
        nlayers=10,
        atm_min_pressure=100.0,
        atm_max_pressure=2e5,
    )

    assert profile.minAtmospherePressure == pytest.approx(100.0)
    assert profile.maxAtmospherePressure == pytest.approx(2e5)


def test_log_pressure_setters_accept_quantities():
    """Fittable pressure setters also normalize Quantity input to Pa."""
    profile = LogPressureProfile()

    profile.minAtmospherePressure = 2.0 * u.mbar
    profile.maxAtmospherePressure = 3.0 * u.bar

    assert profile.minAtmospherePressure == pytest.approx(200.0)
    assert profile.maxAtmospherePressure == pytest.approx(3e5)


@pytest.mark.parametrize("parameter", ["atm_min_pressure", "atm_max_pressure"])
def test_log_pressure_rejects_incompatible_quantity(parameter):
    """Pressure bounds reject quantities with incompatible dimensions."""
    kwargs = {parameter: 1.0 * u.m}

    with pytest.raises(u.UnitConversionError):
        LogPressureProfile(**kwargs)


def test_array_pressure_accepts_quantity():
    """Array pressure profiles are normalized to Pa."""
    profile = ArrayPressureProfile(np.array([1.0, 0.1]) * u.bar)

    np.testing.assert_allclose(profile.profile, [1e5, 1e4])


def test_array_pressure_preserves_unitless_pa_values():
    """Plain arrays retain their legacy interpretation in Pa."""
    profile = ArrayPressureProfile([1e5, 1e4], reverse=True)

    np.testing.assert_allclose(profile.profile, [1e4, 1e5])


def test_array_pressure_computes_levels():
    """The normalized pressure array is used to compute level pressures."""
    profile = ArrayPressureProfile([1e5, 1e4])

    profile.compute_pressure_profile()

    assert profile.pressure_profile_levels.shape == (3,)


def test_array_pressure_rejects_incompatible_quantity():
    """Pressure arrays reject incompatible physical dimensions."""
    with pytest.raises(u.UnitConversionError):
        ArrayPressureProfile(np.array([1.0, 2.0]) * u.m)

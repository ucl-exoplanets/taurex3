"""Test NPoint temperature profile."""

import numpy as np
import pytest
from astropy import units as u
from hypothesis import given
from hypothesis.strategies import floats as st_floats
from hypothesis.strategies import integers as st_integers

from taurex.data.planet import Earth
from taurex.data.profiles.temperature import NPoint
from taurex.exceptions import InvalidModelException

from ..strategies import TP_npoints


@given(
    params=TP_npoints(),
    limit_slope=st_floats(10.0, 999999.0, allow_nan=False),
    smoothing_window=st_integers(1, 100),
)
def test_npoint(params, limit_slope, smoothing_window):
    """Test npoint."""
    nlayers, T_top, T_surface, P_top, P_surface, temp_points, press_points, P = params

    planet = Earth()

    npoint = NPoint(
        T_surface=T_surface,
        T_top=T_top,
        P_surface=P_surface,
        P_top=P_top,
        temperature_points=temp_points,
        pressure_points=press_points,
        limit_slope=limit_slope,
        smoothing_window=smoothing_window,
    )

    # Test params
    npoints = len(temp_points)

    params = npoint.fitting_parameters()

    for x in range(npoints):
        assert f"T_point{x + 1}" in params
        assert params[f"T_point{x + 1}"][2]() == temp_points[x]
        assert f"P_point{x + 1}" in params
        assert params[f"P_point{x + 1}"][2]() == press_points[x]

    npoint.initialize_profile(planet=planet, nlayers=nlayers, pressure_profile=P)

    Pnodes = [P[0], *press_points, P_top]
    Tnodes = [T_surface, *temp_points, T_top]
    diff = np.diff(Tnodes) / np.diff(np.log10(Pnodes))
    if any(Pnodes[i] <= Pnodes[i + 1] for i in range(len(Pnodes) - 1)):

        with pytest.raises(InvalidModelException):
            npoint.profile

    elif any(np.abs(diff) >= limit_slope):
        with pytest.raises(InvalidModelException):
            npoint.profile
    else:
        # Lets make sure it doesn't crash
        npoint.profile


def test_npoint_accepts_quantity_nodes():
    """Temperature and pressure nodes are normalized to K and Pa."""
    profile = NPoint(
        T_surface=1.5 * u.kK,
        T_top=u.Quantity(0.0, u.deg_C),
        P_surface=1.0 * u.bar,
        P_top=1.0 * u.mbar,
        temperature_points=np.array([500.0]) * u.K,
        pressure_points=np.array([0.1]) * u.bar,
    )

    assert profile.temperatureSurface == pytest.approx(1500.0)
    assert profile.temperatureTop == pytest.approx(273.15)
    assert profile.pressureSurface == pytest.approx(1e5)
    assert profile.pressureTop == pytest.approx(100.0)
    np.testing.assert_allclose(profile._t_points, [500.0])
    np.testing.assert_allclose(profile._p_points, [1e4])


def test_npoint_fitting_setters_accept_quantities():
    """Generated node setters normalize Quantity input."""
    profile = NPoint(temperature_points=[500.0], pressure_points=[1e4])
    parameters = profile.fitting_parameters()

    parameters["T_point1"][3](0.6 * u.kK)
    parameters["P_point1"][3](0.2 * u.bar)

    assert parameters["T_point1"][2]() == pytest.approx(600.0)
    assert parameters["P_point1"][2]() == pytest.approx(2e4)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("T_surface", 1.0 * u.m),
        ("T_top", 1.0 * u.m),
        ("P_surface", 1.0 * u.s),
        ("P_top", 1.0 * u.s),
        ("temperature_points", np.array([1.0]) * u.m),
        ("pressure_points", np.array([1.0]) * u.s),
    ],
)
def test_npoint_rejects_incompatible_quantities(parameter, value):
    """NPoint rejects incompatible temperature and pressure dimensions."""
    kwargs = {
        "temperature_points": [500.0] if parameter != "temperature_points" else [],
        "pressure_points": [1e4] if parameter != "pressure_points" else [],
        parameter: value,
    }

    with pytest.raises(u.UnitConversionError):
        NPoint(**kwargs)

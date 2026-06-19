"""Test NPoint temperature profile."""

import numpy as np
import pytest
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

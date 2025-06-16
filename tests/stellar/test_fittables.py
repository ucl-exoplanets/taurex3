"""Test fittable parameters in the Star class."""
import pytest

@pytest.mark.parametrize(
    "param, value, value_set",
    [
        ("star_radius", 0.6, 0.8),
        ("star_mass", 1.2, 1.5),
        ("star_temperature", 5000, 6000),
        ("star_metallicity", 4.0, 5.0),
    ]
)
def test_fittable_bb(param, value, value_set):
    from taurex.stellar import BlackbodyStar
    from taurex.constants import RSOL, MSOL
    star = BlackbodyStar(
        mass = 1.2,
        temperature=5000, radius=0.6, distance=1.0, metallicity=4.0)

    assert star.temperature == 5000
    assert star.radius == 0.6 * RSOL  # RSOL in meters

    assert star.metallicityZ == 4.0
    assert star.mass == 1.2 * MSOL  # MSOL in kg

    assert star.massMsol == 1.2
    assert star.radiusRsol == 0.6

    fittables = star.fitting_parameters()

    assert len(fittables) == 5

    assert "star_radius" in fittables
    assert "star_mass" in fittables
    assert "star_temperature" in fittables
    assert "star_metallicity" in fittables

    radius_param = fittables[param]
    assert radius_param[2]() == value

    # Set to new value
    radius_param[3](value_set)
    assert radius_param[2]() == 0.8



def test_derived_bb():
    from taurex.stellar import BlackbodyStar
    from taurex.constants import RSOL, MSOL
    from astropy import units as u
    from astropy import constants as const
    import numpy as np

    R = 1.0 << u.Rsun
    M = 1.0 << u.Msun

    star = BlackbodyStar(
        mass=1.0, temperature=5000, radius=1.0, distance=1.0, metallicity=4.0
    )

    # Check derived properties
    assert star.radius == 1.0 * RSOL  # in meters
    assert star.mass == 1.0 * MSOL  # in kg
    assert star.metallicityZ == 4.0  # in log scale

    derived = star.derived_parameters()

    assert len(derived) == 1

    assert "star_logg" in derived

    logg = derived["star_logg"]

    g = const.G * (M / R**2)
    logg_value = np.log10((g.to(u.cm / u.s**2)).value)  # Convert to cgs


    assert logg[2]() == pytest.approx(logg_value, rel=1e-8)  # Assuming logg is calculated correctly

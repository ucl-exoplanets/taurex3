"""Test blackbody star."""

import astropy.units as u
import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from ..strategies import hyp_wngrid


def test_blackbody_preserves_plain_value_units():
    """Plain constructor values retain their documented legacy units."""
    from taurex.constants import MSOL
    from taurex.constants import RSOL
    from taurex.stellar import BlackbodyStar

    star = BlackbodyStar(
        temperature=4500,
        radius=2,
        distance=3,
        mass=4,
        metallicity=0.5,
    )

    assert star.temperature == pytest.approx(4500)
    assert star.radius == pytest.approx(2 * RSOL)
    assert star.mass == pytest.approx(4 * MSOL)
    assert star.distance == pytest.approx(3)
    assert star._metallicity == pytest.approx(0.5)


def test_blackbody_accepts_quantity_constructor_values():
    """Quantity constructor values are normalized to internal units."""
    from taurex.stellar import BlackbodyStar

    star = BlackbodyStar(
        temperature=25 * u.deg_C,
        radius=2e6 * u.km,
        distance=10 * u.lyr,
        mass=3 * u.M_earth,
        metallicity=2 * u.percent,
    )

    assert star.temperature == pytest.approx(
        (25 * u.deg_C).to_value(u.K, u.temperature())
    )
    assert star.radius == pytest.approx((2e6 * u.km).to_value(u.m))
    assert star.mass == pytest.approx((3 * u.M_earth).to_value(u.kg))
    assert star.distance == pytest.approx((10 * u.lyr).to_value(u.pc))
    assert star._metallicity == pytest.approx(0.02)


def test_blackbody_quantity_setters():
    """Public stellar setters normalize quantities to their legacy units."""
    from taurex.stellar import BlackbodyStar

    star = BlackbodyStar()

    star.temperature = 3000 * u.K
    star["distance"] = 10 * u.lyr

    assert star.temperature == pytest.approx(3000)
    assert star.distance == pytest.approx((10 * u.lyr).to_value(u.pc))


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("temperature", 1 * u.kg),
        ("radius", 1 * u.s),
        ("distance", 1 * u.K),
        ("mass", 1 * u.m),
        ("metallicity", 1 * u.Pa),
    ],
)
def test_blackbody_rejects_incompatible_quantities(parameter, value):
    """Incompatible stellar quantities raise Astropy conversion errors."""
    from taurex.stellar import BlackbodyStar

    with pytest.raises(u.UnitConversionError):
        BlackbodyStar(**{parameter: value})


@given(
    temperature=st.floats(),
    radius=st.floats(),
    distance=st.floats(),
    magnitudeK=st.floats(),
    mass=st.floats(),
    metallicity=st.floats(),
    wngrid=hyp_wngrid(),
)
@settings(deadline=None)
def test_blackbody_star(
    temperature, radius, distance, magnitudeK, mass, metallicity, wngrid
):
    """Test blackbody star."""
    from taurex.stellar import BlackbodyStar

    bs = BlackbodyStar(temperature, radius, distance, magnitudeK, mass, metallicity)

    bs.initialize(wngrid)

    bs.sed

    assert bs.sed.shape[0] == wngrid.shape[0]

    bs["distance"] = distance + 1

    np.testing.assert_equal(bs.distance, distance + 1)

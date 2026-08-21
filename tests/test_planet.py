"""Tests for planet physical-unit boundaries."""

import astropy.units as u
import pytest

from taurex.planet import Planet


def test_planet_preserves_plain_value_units():
    """Plain constructor values retain their documented legacy units."""
    planet = Planet(planet_mass=2.0, planet_radius=3.0, planet_sma=0.5)

    assert planet.get_planet_mass("Mjup") == pytest.approx(2.0)
    assert planet.get_planet_radius("Rjup") == pytest.approx(3.0)
    assert planet.get_planet_semimajoraxis("AU") == pytest.approx(0.5)


def test_planet_accepts_quantity_constructor_values():
    """Quantity constructor values are normalized to internal SI units."""
    mass = 2.0 * u.M_earth
    radius = 3.0 * u.R_earth
    semimajor_axis = 1.5e6 * u.km

    planet = Planet(
        planet_mass=mass,
        planet_radius=radius,
        planet_sma=semimajor_axis,
    )

    assert planet.get_planet_mass("kg") == pytest.approx(mass.to_value(u.kg))
    assert planet.get_planet_radius("m") == pytest.approx(radius.to_value(u.m))
    assert planet.get_planet_semimajoraxis("m") == pytest.approx(
        semimajor_axis.to_value(u.m)
    )


def test_planet_quantity_uses_attached_unit():
    """A Quantity's attached unit takes precedence over the fallback unit."""
    planet = Planet()

    planet.set_planet_radius(1.0 * u.km, unit="Rjup")

    assert planet.get_planet_radius("m") == pytest.approx(1000.0)


def test_planet_property_setters_accept_quantity_values():
    """Public property setters normalize Quantity values at the API boundary."""
    planet = Planet()

    planet.mass = 2.0 * u.M_earth
    planet.radius = 3.0 * u.R_earth
    planet.distance = 1.5e6 * u.km

    assert planet.get_planet_mass("kg") == pytest.approx((2.0 * u.M_earth).to_value(u.kg))
    assert planet.get_planet_radius("m") == pytest.approx((3.0 * u.R_earth).to_value(u.m))
    assert planet.get_planet_semimajoraxis("m") == pytest.approx((1.5e6 * u.km).to_value(u.m))

    planet.semiMajorAxis = 2.0 * u.AU
    assert planet.get_planet_semimajoraxis("AU") == pytest.approx(2.0)


@pytest.mark.parametrize(
    "setter_name",
    ["set_planet_mass", "set_planet_radius", "set_planet_semimajoraxis"],
)
def test_planet_rejects_incompatible_quantity(setter_name):
    """Planet setters reject quantities with incompatible dimensions."""
    planet = Planet()

    with pytest.raises(u.UnitConversionError):
        getattr(planet, setter_name)(1.0 * u.s)

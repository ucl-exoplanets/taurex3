"""Tests for unit-aware simple and flat cloud inputs."""

import pytest
from astropy import units as u

from taurex.contributions import FlatMieContribution
from taurex.contributions import SimpleCloudsContribution


def test_simple_clouds_constructor_and_setter_normalize_pressure():
    """Simple cloud pressure should be stored as a plain value in Pa."""
    contribution = SimpleCloudsContribution(0.2 * u.bar)

    assert contribution.cloudsPressure == pytest.approx(20_000.0)
    assert not isinstance(contribution.cloudsPressure, u.Quantity)

    parameter = contribution.fitting_parameters()["clouds_pressure"]
    parameter[3](3 * u.mbar)
    assert parameter[2]() == pytest.approx(300.0)


def test_flat_mie_constructor_normalizes_quantities():
    """Flat Mie inputs should be stored in m2 and Pa."""
    contribution = FlatMieContribution(
        flat_mix_ratio=2 * u.cm**2,
        flat_bottomP=0.3 * u.bar,
        flat_topP=2 * u.mbar,
    )

    assert contribution.mieMixing == pytest.approx(2e-4)
    assert contribution.mieBottomPressure == pytest.approx(30_000.0)
    assert contribution.mieTopPressure == pytest.approx(200.0)
    assert not isinstance(contribution.mieMixing, u.Quantity)


@pytest.mark.parametrize(
    ("parameter_name", "quantity", "expected"),
    [
        ("flat_mix_ratio", 3 * u.cm**2, 3e-4),
        ("flat_bottomP", 0.4 * u.bar, 40_000.0),
        ("flat_topP", 5 * u.mbar, 500.0),
    ],
)
def test_flat_mie_fitting_setters_normalize_quantities(
    parameter_name, quantity, expected
):
    """Flat Mie fitting setters should match constructor conversions."""
    parameter = FlatMieContribution().fitting_parameters()[parameter_name]

    parameter[3](quantity)

    assert parameter[2]() == pytest.approx(expected)
    assert not isinstance(parameter[2](), u.Quantity)


@pytest.mark.parametrize(
    ("contribution", "parameter_name", "quantity"),
    [
        (SimpleCloudsContribution(), "clouds_pressure", 1 * u.m),
        (FlatMieContribution(), "flat_mix_ratio", 1 * u.kg),
        (FlatMieContribution(), "flat_bottomP", 1 * u.s),
        (FlatMieContribution(), "flat_topP", 1 * u.K),
    ],
)
def test_remaining_cloud_inputs_reject_incompatible_units(
    contribution, parameter_name, quantity
):
    """Cloud input boundaries should reject incompatible dimensions."""
    parameter = contribution.fitting_parameters()[parameter_name]

    with pytest.raises(u.UnitConversionError):
        parameter[3](quantity)


def test_remaining_cloud_inputs_preserve_legacy_values_and_sentinels():
    """Unitless inputs should retain their existing implicit units."""
    simple = SimpleCloudsContribution(1e4)
    flat = FlatMieContribution(1e-8, -1, -1)

    assert simple.cloudsPressure == 1e4
    assert flat.mieMixing == 1e-8
    assert flat.mieBottomPressure == -1
    assert flat.mieTopPressure == -1

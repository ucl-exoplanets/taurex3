"""Tests for unit-aware Lee Mie inputs."""

import pytest
from astropy import units as u

from taurex.contributions import LeeMieContribution


def test_leemie_constructor_normalizes_quantities():
    """Constructor quantities should be stored in Lee Mie's internal units."""
    contribution = LeeMieContribution(
        lee_mie_radius=100 * u.nm,
        lee_mie_q=40 * u.dimensionless_unscaled,
        lee_mie_mix_ratio=2 / u.cm**3,
        lee_mie_bottomP=0.2 * u.bar,
        lee_mie_topP=1 * u.mbar,
    )

    assert contribution.mieRadius == pytest.approx(0.1)
    assert contribution.mieQ == pytest.approx(40.0)
    assert contribution.mieMixing == pytest.approx(2e6)
    assert contribution.mieBottomPressure == pytest.approx(20_000.0)
    assert contribution.mieTopPressure == pytest.approx(100.0)
    assert not isinstance(contribution.mieRadius, u.Quantity)
    assert not isinstance(contribution.mieMixing, u.Quantity)


@pytest.mark.parametrize(
    ("parameter_name", "quantity", "expected"),
    [
        ("lee_mie_radius", 0.2 * u.mm, 200.0),
        ("lee_mie_q", 25 * u.percent, 0.25),
        ("lee_mie_mix_ratio", 3 / u.cm**3, 3e6),
        ("lee_mie_bottomP", 0.3 * u.bar, 30_000.0),
        ("lee_mie_topP", 2 * u.mbar, 200.0),
    ],
)
def test_leemie_fitting_setters_normalize_quantities(
    parameter_name, quantity, expected
):
    """Fitting setters should apply the same conversions as the constructor."""
    parameter = LeeMieContribution().fitting_parameters()[parameter_name]

    parameter[3](quantity)

    assert parameter[2]() == pytest.approx(expected)
    assert not isinstance(parameter[2](), u.Quantity)


@pytest.mark.parametrize(
    ("parameter_name", "quantity"),
    [
        ("lee_mie_radius", 1 * u.s),
        ("lee_mie_q", 1 * u.kg),
        ("lee_mie_mix_ratio", 1 * u.kg / u.m**3),
        ("lee_mie_bottomP", 1 * u.m),
        ("lee_mie_topP", 1 * u.K),
    ],
)
def test_leemie_rejects_incompatible_units(parameter_name, quantity):
    """Each physical input should reject an incompatible dimension."""
    parameter = LeeMieContribution().fitting_parameters()[parameter_name]

    with pytest.raises(u.UnitConversionError):
        parameter[3](quantity)


def test_leemie_preserves_legacy_values_and_pressure_sentinels():
    """Unitless values should retain their existing implicit units."""
    contribution = LeeMieContribution(0.5, 20, 1e8, -1, -1)

    assert contribution.mieRadius == 0.5
    assert contribution.mieQ == 20
    assert contribution.mieMixing == 1e8
    assert contribution.mieBottomPressure == -1
    assert contribution.mieTopPressure == -1

"""Test Guillot 2010 temperature profile."""

import pytest
from astropy import units as u
from hypothesis import given
from hypothesis import strategies as st

from taurex.data.profiles.temperature import Guillot2010
from taurex.exceptions import InvalidModelException

from ..strategies import planets
from ..strategies import pressures


@given(
    T_irr=st.floats(max_value=1e5, allow_nan=False),
    kappa_ir=st.floats(allow_nan=False),
    kappa_v1=st.floats(allow_nan=False),
    kappa_v2=st.floats(allow_nan=False),
    alpha=st.floats(allow_nan=False),
    T_int=st.floats(max_value=1e5, allow_nan=False),
    P=pressures(),
    planet=planets(),
)
def test_guillot_behaviour(
    T_irr,
    kappa_ir,
    kappa_v1,
    kappa_v2,
    alpha,
    T_int,
    P,
    planet,
):
    """Test guillot behaviour."""
    g = None

    if any(
        [
            kappa_ir == 0.0,
            kappa_v1 == 0.0,
            kappa_v2 == 0.0,
            T_irr < 0,
            T_int < 0,
        ]
    ):
        with pytest.raises(InvalidModelException):
            g = Guillot2010(
                T_irr,
                kappa_ir,
                kappa_v1,
                kappa_v2,
                alpha,
                T_int,
            )
        return
    elif kappa_v1 / kappa_ir == 0.0 or kappa_v2 / kappa_ir == 0.0:
        with pytest.raises(InvalidModelException):
            g = Guillot2010(
                T_irr,
                kappa_ir,
                kappa_v1,
                kappa_v2,
                alpha,
                T_int,
            )
        return
    else:
        g = Guillot2010(
            T_irr,
            kappa_ir,
            kappa_v1,
            kappa_v2,
            alpha,
            T_int,
        )

    nlayers = P.shape[0]

    g.initialize_profile(
        nlayers=nlayers,
        planet=planet,
        pressure_profile=P,
    )

    # Test fitting params
    params = g.fitting_parameters()

    assert "T_irr" in params
    assert "kappa_irr" in params
    assert "kappa_v1" in params
    assert "kappa_v2" in params
    assert "alpha" in params
    assert "T_int_guillot" in params

    assert params["T_irr"][2]() == T_irr
    assert params["kappa_irr"][2]() == kappa_ir
    assert params["kappa_v1"][2]() == kappa_v1
    assert params["kappa_v2"][2]() == kappa_v2
    assert params["alpha"][2]() == alpha
    assert params["T_int_guillot"][2]() == T_int

    g.profile

    # Test zeroing behaviour
    try:
        old_value = params["kappa_irr"][2]()
        params["kappa_irr"][3](0.0)
        g.profile
        raise AssertionError("Should have raised InvalidModelException")
    except InvalidModelException:
        params["kappa_irr"][3](old_value)
        assert True

    try:
        old_value = params["kappa_v1"][2]()
        params["kappa_v1"][3](0.0)
        g.profile
        raise AssertionError("Should have raised InvalidModelException")
    except InvalidModelException:
        params["kappa_v1"][3](old_value)
        assert True

    try:
        old_value = params["kappa_v2"][2]()
        params["kappa_v2"][3](0.0)
        g.profile
        raise AssertionError("Should have raised InvalidModelException")
    except InvalidModelException:
        params["kappa_v2"][3](old_value)
        assert True


def test_guillot_values():
    """Test guillot values.

    Should be a list of inputs and outputs
    """
    pass


def test_guillot_accepts_temperature_quantities():
    """Physical temperature parameters are normalized to Kelvin."""
    profile = Guillot2010(
        T_irr=1.5 * u.kK,
        T_int=u.Quantity(20.0, u.deg_C),
    )

    assert profile.equilTemperature == pytest.approx(1500.0)
    assert profile.internalTemperature == pytest.approx(293.15)


def test_guillot_temperature_setters_accept_quantities():
    """Fittable temperature setters normalize Quantity values."""
    profile = Guillot2010()

    profile.equilTemperature = 2.0 * u.kK
    profile.internalTemperature = 0.5 * u.kK

    assert profile.equilTemperature == pytest.approx(2000.0)
    assert profile.internalTemperature == pytest.approx(500.0)


@pytest.mark.parametrize("parameter", ["T_irr", "T_int"])
def test_guillot_rejects_incompatible_temperature_quantity(parameter):
    """Temperature parameters reject incompatible physical dimensions."""
    with pytest.raises(u.UnitConversionError):
        Guillot2010(**{parameter: 1.0 * u.m})


def test_guillot_rejects_quantity_below_absolute_zero():
    """Validation occurs after conversion to the internal Kelvin unit."""
    with pytest.raises(InvalidModelException, match="Negative temperature"):
        Guillot2010(T_int=u.Quantity(-300.0, u.deg_C))


def test_guillot_accepts_opacity_and_ratio_quantities():
    """Opacity and ratio inputs are normalized to their internal units."""
    profile = Guillot2010(
        kappa_irr=100.0 * u.cm**2 / u.g,
        kappa_v1=50.0 * u.cm**2 / u.g,
        kappa_v2=25.0 * u.cm**2 / u.g,
        alpha=50.0 * u.percent,
    )

    assert profile.meanInfraOpacity == pytest.approx(10.0)
    assert profile.meanOpticalOpacity1 == pytest.approx(5.0)
    assert profile.meanOpticalOpacity2 == pytest.approx(2.5)
    assert profile.opticalRatio == pytest.approx(0.5)
    assert not isinstance(profile.kappa_ir, u.Quantity)
    assert not isinstance(profile.alpha, u.Quantity)


def test_guillot_opacity_and_ratio_setters_accept_quantities():
    """Fittable opacity and ratio setters normalize Quantity values."""
    profile = Guillot2010()
    parameters = profile.fitting_parameters()

    parameters["kappa_irr"][3](20.0 * u.cm**2 / u.g)
    parameters["kappa_v1"][3](30.0 * u.cm**2 / u.g)
    parameters["kappa_v2"][3](40.0 * u.cm**2 / u.g)
    parameters["alpha"][3](25.0 * u.percent)

    assert profile.meanInfraOpacity == pytest.approx(2.0)
    assert profile.meanOpticalOpacity1 == pytest.approx(3.0)
    assert profile.meanOpticalOpacity2 == pytest.approx(4.0)
    assert profile.opticalRatio == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("parameter", "quantity"),
    [
        ("kappa_irr", 1.0 * u.K),
        ("kappa_v1", 1.0 * u.m),
        ("kappa_v2", 1.0 * u.kg),
        ("alpha", 1.0 * u.Pa),
    ],
)
def test_guillot_rejects_incompatible_physical_quantities(parameter, quantity):
    """Opacity and ratio parameters reject incompatible dimensions."""
    with pytest.raises(u.UnitConversionError):
        Guillot2010(**{parameter: quantity})

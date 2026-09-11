"""Tests for unit-aware gas-profile input boundaries."""

import numpy as np
import pytest
from astropy import units as u

from taurex.chemistry import CustomGas
from taurex.chemistry import PowerGas
from taurex.chemistry import TwoLayerGas
from taurex.chemistry import TwoPointGas


@pytest.mark.parametrize(
    ("gas", "parameter_names"),
    [
        (PowerGas("H2O", mix_ratio_surface=1e-4), ("H2O_surface",)),
        (
            TwoLayerGas("CH4"),
            ("CH4_surface", "CH4_top"),
        ),
        (TwoPointGas("CO"), ("CO_surface", "CO_top")),
    ],
)
def test_gas_fitting_setters_normalize_dimensionless_quantities(gas, parameter_names):
    """Dynamic fitting setters should accept dimensionless quantities."""
    parameters = gas.fitting_parameters()

    for parameter_name in parameter_names:
        parameters[parameter_name][3](5 * u.percent)
        assert parameters[parameter_name][2]() == pytest.approx(0.05)


def test_two_layer_pressure_fitting_setter_normalizes_quantity():
    """The dynamic boundary-pressure setter should store Pascals."""
    gas = TwoLayerGas("CH4")
    parameter = gas.fitting_parameters()["CH4_P"]

    parameter[3](0.2 * u.bar)

    assert parameter[2]() == pytest.approx(20_000.0)


def test_custom_gas_normalizes_quantity_array_and_fitting_setter():
    """The custom gas should normalize constructor and fitting-setter arrays."""
    gas = CustomGas("H2O", np.array([5, 1]) * u.percent)
    gas.initialize_profile(nlayers=2)

    np.testing.assert_allclose(gas.mixProfile, [0.05, 0.01])
    assert not isinstance(gas.mixProfile, u.Quantity)

    gas.fitting_parameters()["H2O"][3](np.array([2, 1]) * u.percent)
    gas.initialize_profile(nlayers=2)
    np.testing.assert_allclose(gas.mixProfile, [0.02, 0.01])


@pytest.mark.parametrize(
    "gas",
    [
        CustomGas("H2O", [1e-4]),
        PowerGas("H2O", mix_ratio_surface=1e-4),
        TwoLayerGas("CH4"),
        TwoPointGas("CO"),
    ],
)
def test_gas_quantity_boundaries_reject_physical_units(gas):
    """VMR inputs should reject quantities with physical dimensions."""
    parameter_name = next(iter(gas.fitting_parameters()))

    with pytest.raises(u.UnitConversionError):
        gas.fitting_parameters()[parameter_name][3](1 * u.kg)

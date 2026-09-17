"""Tests for constant gas."""

import numpy as np
from astropy import units as u
from hypothesis import given
from hypothesis.strategies import floats
from hypothesis.strategies import integers

from taurex.chemistry import ConstantGas
from taurex.data.profiles.chemistry.gas.arraygas import ArrayGas

from ..strategies import molecule_vmr


@given(molecule=molecule_vmr(), nlayers=integers(1, 300), new_value=floats(1e-30, 1e0))
def test_constant_gas(molecule, nlayers, new_value):
    """Test constant gas profile."""
    mol, vmr = molecule
    cg = ConstantGas(mol[0], mix_ratio=vmr)
    cg.initialize_profile(nlayers=nlayers)

    mix_profile = cg.mixProfile

    assert np.all(mix_profile == vmr)

    params = cg.fitting_parameters()

    assert mol[0] in params

    params = params[mol[0]]

    name = params[0]
    getter = params[2]
    setter = params[3]

    assert name == mol[0]
    assert getter() == vmr
    setter(new_value)
    assert getter() == new_value

    cg.initialize_profile(nlayers=nlayers)
    mix_profile = cg.mixProfile
    assert np.all(mix_profile == new_value)


def test_constant_gas_quantity_input():
    """Dimensionless quantities should be normalized to plain numeric VMR data."""
    cg = ConstantGas("H2O", mix_ratio=5 * u.percent)

    assert not isinstance(cg._mix_ratio, u.Quantity)
    assert np.isclose(cg._mix_ratio, 0.05)
    cg.initialize_profile(nlayers=3)
    assert np.allclose(cg.mixProfile, 0.05)


def test_array_gas_quantity_sequence_input():
    """Normalize an ArrayGas sequence of dimensionless quantities."""
    gas = ArrayGas("H2O", mix_ratio_array=np.array([5, 1]) * u.percent)

    assert not np.any(
        np.vectorize(lambda x: isinstance(x, u.Quantity))(gas._mix_ratio_array)
    )
    np.testing.assert_allclose(gas._mix_ratio_array, [0.05, 0.01])

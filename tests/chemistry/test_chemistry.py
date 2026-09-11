"""Tests for chemistry."""

from unittest.mock import patch

import numpy as np
import pytest
from astropy import units as u
from hypothesis import given
from hypothesis.strategies import lists

from taurex.chemistry import Chemistry

from ..strategies import molecules


def setup_active(active_molecules, inactive_molecules):
    """Setup active and inactive molecules."""
    from taurex.cache import OpacityCache
    from taurex.cache.ktablecache import KTableCache

    with patch.object(OpacityCache, "find_list_of_molecules") as mock_my_method_xsec:
        with patch.object(KTableCache, "find_list_of_molecules") as mock_my_method_ktab:
            mock_my_method_xsec.return_value = active_molecules
            mock_my_method_ktab.return_value = inactive_molecules
            c = Chemistry("test")

    return c


@given(mols=lists(molecules()))
def test_chemistry_active_default(mols):
    """Test active default."""
    from taurex.cache import GlobalCache

    active_molecules = [m[0] for m in mols]
    inactive_molecules = []
    num_molecules = len(active_molecules)

    if num_molecules > 1:
        active_molecules = active_molecules[: num_molecules // 2]
        inactive_molecules = active_molecules[num_molecules // 2 :]

    gc = GlobalCache()
    if "opacity_method" in gc.variable_dict:
        del gc.variable_dict["opacity_method"]

    c = setup_active(active_molecules, inactive_molecules)

    if len(mols) == 0:
        assert len(c.availableActive) == 0
    else:

        assert c.availableActive == active_molecules


@given(mols=lists(molecules()))
def test_chemistry_active_xsec(mols):
    """Test active xsec."""
    from taurex.cache import GlobalCache

    active_molecules = [m[0] for m in mols]
    inactive_molecules = []
    num_molecules = len(active_molecules)

    if num_molecules > 1:
        active_molecules = active_molecules[: num_molecules // 2]
        inactive_molecules = active_molecules[num_molecules // 2 :]

    gc = GlobalCache()
    gc["opacity_method"] = "xsec"

    c = setup_active(active_molecules, inactive_molecules)

    if len(mols) == 0:
        assert len(c.availableActive) == 0
    else:

        assert c.availableActive == active_molecules


@given(mols=lists(molecules()))
def test_chemistry_active_ktable(mols):
    """Test active ktable."""
    from taurex.cache import GlobalCache

    active_molecules = [m[0] for m in mols]
    inactive_molecules = []
    num_molecules = len(active_molecules)

    if num_molecules > 1:
        active_molecules = active_molecules[: num_molecules // 2]
        inactive_molecules = active_molecules[num_molecules // 2 :]

    gc = GlobalCache()
    gc["opacity_method"] = "ktables"

    c = setup_active(inactive_molecules, active_molecules)

    if len(mols) == 0:
        assert len(c.availableActive) == 0
    else:

        assert c.availableActive == active_molecules


def test_normalize_condensate_number_density_quantity():
    """Number-density quantities are normalized to plain values in m^-3."""
    values = Chemistry.normalize_condensate_number_density(
        np.array([1.0, 2.0]) / u.cm**3
    )

    np.testing.assert_allclose(values, [1.0e6, 2.0e6])
    assert not isinstance(values, u.Quantity)


def test_normalize_condensate_number_density_preserves_plain_values():
    """Unitless values retain their existing implicit m^-3 meaning."""
    values = np.array([1.0, 2.0])

    assert Chemistry.normalize_condensate_number_density(values) is values


def test_normalize_condensate_number_density_rejects_mass_density():
    """Mass density is not silently interpreted as particle number density."""
    with pytest.raises(u.UnitConversionError):
        Chemistry.normalize_condensate_number_density(1.0 * u.kg / u.m**3)


def test_get_condensate_number_density_profile():
    """The named accessor selects a profile using the condensate ordering."""

    class NumberDensityChemistry(Chemistry):
        @property
        def condensates(self):
            return ["MgSiO3", "Fe"]

        @property
        def condensateNumberDensityProfile(self):  # noqa: N802
            return np.array([[1.0, 2.0], [3.0, 4.0]])

    chemistry = object.__new__(NumberDensityChemistry)

    np.testing.assert_allclose(
        chemistry.get_condensate_number_density_profile("Fe"), [3.0, 4.0]
    )
    with pytest.raises(KeyError, match="Unknown"):
        chemistry.get_condensate_number_density_profile("Unknown")


def test_number_density_accessor_rejects_unsupported_representation():
    """The named accessor reports when chemistry only provides mix profiles."""

    class MixChemistry(Chemistry):
        @property
        def condensates(self):
            return ["MgSiO3"]

    chemistry = object.__new__(MixChemistry)

    with pytest.raises(ValueError, match="does not provide"):
        chemistry.get_condensate_number_density_profile("MgSiO3")

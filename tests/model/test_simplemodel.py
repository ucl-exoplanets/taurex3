"""Tests for the one-dimensional forward model."""

import astropy.units as u
import numpy as np
import pytest

from taurex.model.simplemodel import OneDForwardModel
from taurex.types import get_float_dtype


@pytest.fixture
def model():
    """Create a minimal model for testing native-grid normalization."""
    instance = object.__new__(OneDForwardModel)
    instance._native_grid = None
    return instance


def test_set_native_grid_accepts_plain_array(model):
    """Plain arrays remain supported and are stored sorted as floats."""
    model.set_native_grid(np.array([3, 1, 2]))

    np.testing.assert_array_equal(model.nativeWavenumberGrid, [1.0, 2.0, 3.0])
    assert model.nativeWavenumberGrid.dtype == get_float_dtype()


def test_set_native_grid_accepts_wavenumber_quantity(model):
    """Wavenumber quantities are normalized to the internal unit."""
    grid = np.array([3.0, 1.0, 2.0]) * u.k

    model.set_native_grid(grid)

    np.testing.assert_allclose(model.nativeWavenumberGrid, [1.0, 2.0, 3.0])


def test_set_native_grid_converts_wavelength_quantity(model):
    """Wavelength quantities are converted using spectral equivalencies."""
    grid = np.array([1.0, 2.0, 3.0]) * u.micron

    model.set_native_grid(grid)

    expected = np.sort(grid.to(u.k, equivalencies=u.spectral()).value)
    np.testing.assert_allclose(model.nativeWavenumberGrid, expected)


def test_set_native_grid_rejects_incompatible_quantity(model):
    """Quantities that are not spectral coordinates are rejected."""
    with pytest.raises(u.UnitConversionError):
        model.set_native_grid(np.array([1.0, 2.0, 3.0]) * u.kg)

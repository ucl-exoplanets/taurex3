"""Test binning."""

import numpy as np
import pytest

from taurex.binning import Binner
from taurex.binning import FluxBinner
from taurex.binning import NativeBinner
from taurex.binning import SimpleBinner
from taurex.binning.fluxbinner import FluxBinnerOld


def test_binner():
    """Test binner."""
    b = Binner()
    with pytest.raises(NotImplementedError):
        b.bindown(None, None)


def test_fluxbinner(spectra):
    """Test fluxbinner."""
    wngrid = np.linspace(400, 20000, 100)
    fb = FluxBinner(wngrid=wngrid)
    wn, sp, _, _ = fb.bindown(*spectra)

    assert wngrid.shape[0] == wn.shape[0]

    assert np.mean(sp) == pytest.approx(np.mean(spectra[1]), rel=0.1)


def test_fluxbinner_retrocompatibility():
    """Test that FluxBinner behaves like FluxBinnerOld."""
    bins = np.array([5000, 6000, 7000])
    bin_widths = np.array([100, 200, 300])
    binner = FluxBinner(bins, wngrid_width=bin_widths)
    binner_old = FluxBinnerOld(bins, wngrid_width=bin_widths)

    wngrid = 1e4 / np.logspace(-0.5, 1, 100000)
    spec = np.random.rand(100000)

    bin_wn, bin_spec, _, _ = binner.bindown(wngrid, spec)
    bin_wn, bin_spec_old, _, _ = binner_old.bindown(wngrid, spec)

    assert bin_spec == pytest.approx(bin_spec_old, rel=1e-6)


def test_fluxbinner_linearity_wn():
    """Test that FluxBinner is linear in wavenumber space."""
    bins = np.array([5000, 6000, 7000])
    bin_widths = np.array([100, 200, 300])
    binner = FluxBinner(bins, wngrid_width=bin_widths)

    wngrid = 1e4 / np.logspace(-0.5, 1, 100000)
    spec = wngrid  # spec is linear in wn space

    bin_wn, bin_spec, _, _ = binner.bindown(wngrid, spec)

    assert bin_spec == pytest.approx(bins, rel=1e-8)


def test_fluxbinner_linearity_wl():
    """Test that FluxBinner is linear in wavenumber space."""
    bins = np.array([2, 3, 4])
    bin_widths = np.array([0.1, 0.2, 0.3])
    binner = FluxBinner(wlgrid=bins, wlgrid_width=bin_widths)

    wngrid = 1e4 / np.logspace(-0.5, 1, 100000)
    spec = 1e4 / wngrid  # spec is linear in wl space

    bin_wl, bin_spec, _, _ = binner.bindown(wngrid, spec)

    assert bin_spec == pytest.approx(bins, rel=1e-8)


def test_simplebinner(spectra):
    """Test simplebinner."""
    wngrid = np.linspace(400, 20000, 100)
    fb = SimpleBinner(wngrid=wngrid)
    wn, sp, _, _ = fb.bindown(*spectra)

    assert wngrid.shape[0] == wn.shape[0]

    assert np.mean(sp) == pytest.approx(np.mean(spectra[1]), rel=0.1)


def test_native_binner():
    """Test native binning."""
    nb = NativeBinner()

    wngrid, spectra = np.linspace(1, 100, 100), np.random.rand(100)

    res = nb.bindown(wngrid, spectra)

    np.testing.assert_equal(wngrid, res[0])
    np.testing.assert_equal(spectra, res[1])

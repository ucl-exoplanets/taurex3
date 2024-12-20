import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import booleans

from taurex.binning import Binner, FluxBinner, NativeBinner, SimpleBinner

from .strategies import wngrid_spectra


def test_binner():
    b = Binner()
    with pytest.raises(NotImplementedError):
        b.bindown(None, None)


def test_fluxbinner(spectra):
    wngrid = np.linspace(400, 20000, 100)
    fb = FluxBinner(wngrid=wngrid)
    wn, sp, _, _ = fb.bindown(*spectra)

    assert wngrid.shape[0] == wn.shape[0]

    assert np.mean(sp) == pytest.approx(np.mean(spectra[1]), rel=0.1)


def test_simplebinner(spectra):
    wngrid = np.linspace(400, 20000, 100)
    fb = SimpleBinner(wngrid=wngrid)
    wn, sp, _, _ = fb.bindown(*spectra)

    assert wngrid.shape[0] == wn.shape[0]

    assert np.mean(sp) == pytest.approx(np.mean(spectra[1]), rel=0.1)

def test_native_binner():
    """Test native binning."""

    

    nb = NativeBinner()

    wngrid, spectra = np.linspace(1,100,100), np.random.rand(100)


    res = nb.bindown(wngrid, spectra)

    np.testing.assert_equal(wngrid, res[0])
    np.testing.assert_equal(spectra, res[1])
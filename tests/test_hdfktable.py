"""Tests for HDF5KTable."""

import numpy as np
import pytest

from taurex.opacity.ktables.hdfktable import HDF5KTable


@pytest.fixture
def hdf5_ktable_path(tmp_path):
    """Create a minimal HDF5 ktable file."""
    import h5py

    n_t = 5
    n_p = 4
    n_wav = 20
    n_gauss = 2

    t_grid = np.linspace(300.0, 1000.0, n_t)
    p_grid = np.logspace(1.0, 6.0, n_p)
    bin_centers = np.linspace(500.0, 3000.0, n_wav)
    weights = np.array([0.5, 0.5])
    kcoeff = np.random.rand(n_p, n_t, n_wav, n_gauss)

    path = tmp_path / "H2O_test.h5"
    with h5py.File(path, "w") as f:
        f["t"] = t_grid
        p = f.create_dataset("p", data=p_grid)
        p.attrs["units"] = "Pa"
        f["bin_centers"] = bin_centers
        f["ngauss"] = n_gauss
        f["weights"] = weights
        f["kcoeff"] = kcoeff

    return path


def test_hdfktable_loads_as_numpy_arrays(hdf5_ktable_path):
    """HDF5KTable should load datasets as numpy arrays, not h5py Datasets."""
    ktable = HDF5KTable(hdf5_ktable_path)

    assert isinstance(ktable.wavenumberGrid, np.ndarray)
    assert isinstance(ktable.temperatureGrid, np.ndarray)
    assert isinstance(ktable.pressureGrid, np.ndarray)
    assert isinstance(ktable.xsecGrid, np.ndarray)
    assert isinstance(ktable.weights, np.ndarray)

"""Array spectra and numerical compatibility with the HDF5 adapter."""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest

from taurex.plot import plot_spectrum


def assert_bands(bands, wavelengths, values, std):
    """Check uncertainty bounds at each wavelength."""
    assert len(bands) == 2
    for sigma, band in enumerate(bands, start=1):
        vertices = band.get_paths()[0].vertices
        for wl, value, error in zip(wavelengths, values, std, strict=True):
            bounds = vertices[vertices[:, 0] == wl, 1]
            np.testing.assert_allclose(
                [bounds.min(), bounds.max()],
                [value - sigma * error, value + sigma * error],
                rtol=1e-12,
            )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("uncertainty", [False, True])
def test_spectrum_values(reverse, uncertainty):
    """Retain irregular coordinates, signed values, styles and uncertainty bounds."""
    wavelengths = np.array([0.5, 1.0, 7.0, 10.0])
    values = np.array([-0.1, 0.0, 1e-14, 0.5])
    std = np.array([0.2, 0.1, 0.0, 0.05])
    if reverse:
        wavelengths, values, std = (v[::-1] for v in (wavelengths, values, std))
    fig, (other, ax) = plt.subplots(1, 2)
    try:
        plt.sca(other)
        line, bands = plot_spectrum(
            ax,
            wavelengths,
            values,
            std=std if uncertainty else None,
            color="red",
            label="Model",
            linestyle="--",
            alpha=0.7,
        )
        np.testing.assert_array_equal(line.get_xdata(), wavelengths)
        np.testing.assert_array_equal(line.get_ydata(), values)
        assert line.get_color() == "red"
        assert line.get_label() == "Model"
        assert line.get_linestyle() == "--"
        assert line.get_alpha() == 0.7
        assert not other.lines and not other.collections
        assert ax.get_xscale() == ax.get_yscale() == "linear"
        if uncertainty:
            assert_bands(bands, wavelengths, values, std)
            for band, alpha, zorder in zip(bands, (0.5, 0.2), (-2, -3), strict=True):
                np.testing.assert_allclose(band.get_facecolor()[0], [1, 0, 0, alpha])
                assert band.get_zorder() == zorder
        else:
            assert bands == ()
        fig.canvas.draw()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("std", [None, [0.0], [0.1]])
def test_single_sample(std):
    """Allow a single wavelength sample."""
    fig, ax = plt.subplots()
    try:
        line, bands = plot_spectrum(ax, [2.0], [0.0], std=std)
        np.testing.assert_array_equal(line.get_ydata(), [0.0])
        assert len(bands) == (0 if std is None else 2)
        fig.canvas.draw()
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "wavelengths,values,std",
    [
        ([], [], None),
        ([[1.0]], [1.0], None),
        ([0.0], [1.0], None),
        ([np.nan], [1.0], None),
        ([np.inf], [1.0], None),
        ([1.0], [1.0, 2.0], None),
        ([1.0], [[1.0]], None),
        ([1.0], [np.nan], None),
        ([1.0], [np.inf], None),
        ([1.0], [1.0], [-1.0]),
        ([1.0], [1.0], [np.nan]),
        ([1.0], [1.0], [np.inf]),
        ([1.0], [1.0], [1.0, 2.0]),
    ],
)
def test_invalid_spectrum(wavelengths, values, std):
    """Reject inconsistent arrays before modifying the axes."""
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError):
            plot_spectrum(ax, wavelengths, values, std=std)
        assert not ax.lines and not ax.collections
    finally:
        plt.close(fig)


@pytest.mark.parametrize("key", ["binned_spectrum", "bin_spectrum", "native_spectrum"])
@pytest.mark.parametrize("uncertainty", [False, True])
def test_hdf5_spectrum_values(key, uncertainty, tmp_path):
    """Preserve legacy positive-value selection and all supported dataset names."""
    from taurex.plot.plotter import Plotter

    wavelengths = np.array([10.0, 7.0, 1.0, 0.5])
    values = np.array([0.5, 0.0, -0.1, 1e-14])
    errors = np.array([0.05, 0.1, 0.2, 0.0])
    fig, ax = plt.subplots()
    try:
        with h5py.File(tmp_path / "spectrum.h5", "w") as handle:
            handle[key] = values
            if uncertainty:
                handle["binned_std"] = errors
            plotter = Plotter.__new__(Plotter)
            line, bands = plotter._generic_plot(
                wavelengths,
                10000 / wavelengths,
                handle,
                None,
                color="red",
                alpha=0.8,
                label="Fit",
            )
        good = values > 0
        np.testing.assert_array_equal(line.get_xdata(), wavelengths[good])
        np.testing.assert_array_equal(line.get_ydata(), values[good])
        assert line.get_color() == "red"
        if uncertainty:
            assert_bands(bands, wavelengths[good], values[good], errors[good])
        else:
            assert bands == ()
        fig.canvas.draw()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("uncertainty", [False, True])
def test_rebinned_spectrum(uncertainty, tmp_path):
    """The adapter retains its existing FluxBinner grid and numerical results."""
    from taurex.binning import FluxBinner
    from taurex.plot.plotter import Plotter
    from taurex.util.util import create_grid_res
    from taurex.util.util import wnwidth_to_wlwidth

    native_grid = np.linspace(500.0, 25000.0, 1000)
    native_spectrum = 0.01 + 0.001 * np.sin(native_grid / 1000)
    native_std = np.full(1000, 0.0001)
    wavelengths = np.array([1.0, 2.0, 5.0])
    grid = create_grid_res(10, 0.9, 5.5)
    order = (10000 / grid[:, 0]).argsort()
    expected_wl = grid[order, 0]
    binner = FluxBinner(
        10000 / expected_wl, wnwidth_to_wlwidth(expected_wl, grid[order, 1])
    )
    expected = binner.bindown(native_grid, native_spectrum)[1]
    expected_std = binner.bindown(native_grid, native_std)[1]
    fig, ax = plt.subplots()
    try:
        with h5py.File(tmp_path / "native.h5", "w") as handle:
            handle["native_spectrum"] = native_spectrum
            if uncertainty:
                handle["native_std"] = native_std
            line, bands = Plotter.__new__(Plotter)._generic_plot(
                wavelengths, native_grid, handle, 10
            )
        np.testing.assert_array_equal(line.get_xdata(), expected_wl)
        np.testing.assert_allclose(line.get_ydata(), expected, rtol=1e-12)
        if uncertainty:
            assert_bands(bands, expected_wl, expected, expected_std)
        else:
            assert bands == ()
        fig.canvas.draw()
    finally:
        plt.close(fig)

"""Standalone depth plots and compatibility with the HDF5 plotter."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from taurex.plot import plot_depth_contrib
from taurex.plot import plot_optical_depth


matplotlib.use("Agg")


@pytest.mark.parametrize("reverse_pressure", [False, True])
@pytest.mark.parametrize("reverse_wavelength", [False, True])
def test_depth_plots(reverse_pressure, reverse_wavelength):
    """Real coordinates retain the correspondence between layers and spectra."""
    pressure = np.array([1e5, 1e3, 1.0])
    wavelengths = np.array([0.5, 2.0, 10.0, 11.0])
    tau = np.arange(12).reshape(3, 4)
    if reverse_pressure:
        pressure, tau = pressure[::-1], tau[::-1]
    if reverse_wavelength:
        wavelengths, tau = wavelengths[::-1], tau[:, ::-1]
    fig, (ax, profile) = plt.subplots(1, 2)
    try:
        mesh = plot_optical_depth(ax, tau, pressure, wavelengths, cmap="magma")
        line = plot_depth_contrib(profile, tau, pressure, color="red")
        np.testing.assert_array_equal(mesh.get_array(), tau)
        coords = mesh.get_coordinates()
        assert np.all(coords > 0)
        edges = coords[0, :, 0]
        assert np.all(wavelengths > np.minimum(edges[:-1], edges[1:]))
        assert np.all(wavelengths < np.maximum(edges[:-1], edges[1:]))
        np.testing.assert_allclose(line.get_ydata(), pressure / 1e5)
        np.testing.assert_allclose(line.get_xdata(), tau.mean(axis=1))
        assert mesh.cmap.name == "magma"
        assert line.get_color() == "red"
        assert ax.get_ylim() == profile.get_ylim() == (1.0, 1e-5)
        assert ax.get_xscale() == ax.get_yscale() == "log"
        assert len(fig.axes) == 2
        fig.canvas.draw()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("pressure", [[0, 1], [np.nan, 1], [1, 1], [[1, 2]], [1, 3, 2]])
def test_invalid_pressure(pressure):
    """Reject pressure grids that cannot describe a depth axis."""
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="pressure"):
            plot_depth_contrib(ax, np.ones((len(pressure), 3)), pressure)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("wavelengths", [[1], [1, 0], [1, np.inf], [2, 2]])
def test_invalid_wavelengths(wavelengths):
    """Reject inconsistent or invalid spectral grids."""
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="wavelength"):
            plot_optical_depth(ax, np.ones((2, 2)), [1, 10], wavelengths)
    finally:
        plt.close(fig)


def test_plotter_depth_layout():
    """The existing plotter composes standalone plots without opening HDF5."""
    from taurex.plot.plotter import Plotter

    plotter = Plotter.__new__(Plotter)
    try:
        plotter._plot_tau(np.ones((3, 4)), [1e5, 1e3, 1], [10, 4, 2, 1])
        fig = plt.gcf()
        assert len(fig.axes) == 3
        fig.canvas.draw()
    finally:
        plt.close("all")


@pytest.mark.parametrize("reverse_pressure", [False, True])
@pytest.mark.parametrize("reverse_wavelength", [False, True])
@pytest.mark.parametrize("wavelengths", [[0.5, 2.0, 8.0], [0.5, 7.0, 8.0]])
def test_plotter_preserves_legacy_values(
    reverse_pressure, reverse_wavelength, wavelengths
):
    """Preserve legacy plotted values independently of rendering and grid order."""
    from taurex.plot.plotter import Plotter

    # Numerical baseline verified against _plot_tau at b0768b46903559798f9d69682e48e259e8f2575f.
    # Keep the expected values here so tests also work without Git history.
    pressure = np.array([1e5, 3e3, 10.0, 1.0])
    wavelengths = np.array(wavelengths)
    tau = np.array(
        [[0.0, 0.0, 0.0], [0.125, 0.25, 0.375], [1.0, 2.0, 7.5], [1e-14, 2e-14, 3e-14]]
    )
    expected_profile = np.array([0.0, 0.25, 3.5, 2e-14])
    expected_pressure = np.array([1.0, 0.03, 1e-4, 1e-5])
    if reverse_pressure:
        pressure, tau = pressure[::-1], tau[::-1]
        expected_profile = expected_profile[::-1]
        expected_pressure = expected_pressure[::-1]
    if reverse_wavelength:
        wavelengths, tau = wavelengths[::-1], tau[:, ::-1]

    plotter = Plotter.__new__(Plotter)
    fig = plt.figure("Contribution function")
    try:
        plotter._plot_tau(tau, pressure, wavelengths)
        main, _, profile = fig.axes
        np.testing.assert_array_equal(main.collections[0].get_array(), tau)
        np.testing.assert_allclose(
            profile.lines[0].get_xdata(),
            expected_profile,
            rtol=1e-12,
            atol=0,
            equal_nan=False,
        )
        np.testing.assert_allclose(
            profile.lines[0].get_ydata(),
            expected_pressure,
            rtol=1e-12,
            atol=0,
            equal_nan=False,
        )
    finally:
        plt.close(fig)

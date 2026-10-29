"""Named array contributions and HDF5 selection compatibility."""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest

from taurex.plot import plot_spectral_contributions


@pytest.mark.parametrize("reverse", [False, True])
def test_array_contributions(reverse):
    """Names, signed values, per-component errors and default colors survive."""
    grid = np.array([0.5, 2.0, 7.0])
    values = {
        "Absorption-H2O": np.array([0.0, -0.1, 0.2]),
        "Rayleigh": np.array([0.3, 0.2, 0.1]),
    }
    errors = np.array([0.01, 0.02, 0.03])
    if reverse:
        grid = grid[::-1]
        values = {name: array[::-1] for name, array in values.items()}
        errors = errors[::-1]
    fig, (other, ax) = plt.subplots(1, 2)
    try:
        plt.sca(other)
        artists = plot_spectral_contributions(
            ax,
            grid,
            values,
            std={"Absorption-H2O": errors},
            linestyle="--",
        )
        assert list(artists) == list(values)
        assert not other.lines and not other.collections
        for name, (line, bands) in artists.items():
            np.testing.assert_array_equal(line.get_xdata(), grid)
            np.testing.assert_array_equal(line.get_ydata(), values[name])
            assert line.get_label() == name
            assert line.get_linestyle() == "--"
            if name == "Rayleigh":
                assert bands == ()
            else:
                assert len(bands) == 2
                for sigma, band in enumerate(bands, start=1):
                    vertices = band.get_paths()[0].vertices
                    for wl, value, error in zip(
                        grid, values[name], errors, strict=True
                    ):
                        bounds = vertices[vertices[:, 0] == wl, 1]
                        np.testing.assert_allclose(
                            [bounds.min(), bounds.max()],
                            [value - sigma * error, value + sigma * error],
                        )
        assert ax.lines[0].get_color() != ax.lines[1].get_color()
        assert ax.get_xscale() == ax.get_yscale() == "linear"
        fig.canvas.draw()
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "values,std,kwargs",
    [
        ({"a": [1.0, 2.0], "b": [1.0]}, None, {}),
        ({"a": [1.0, 2.0]}, {"b": [0.0, 0.0]}, {}),
        ({"a": [1.0, 2.0], "b": [1.0, 2.0]}, {"b": [-1.0, 0.0]}, {}),
        ({"a": [1.0, 2.0]}, None, {"label": "override"}),
    ],
)
def test_invalid_contributions(values, std, kwargs):
    """Validate all components before drawing any of them."""
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError):
            plot_spectral_contributions(ax, [1.0, 2.0], values, std=std, **kwargs)
        assert not ax.lines and not ax.collections
    finally:
        plt.close(fig)


def test_empty_contributions():
    """An empty selection leaves the axes untouched."""
    fig, ax = plt.subplots()
    try:
        assert plot_spectral_contributions(ax, [1.0], {}) == {}
        assert not ax.lines and not ax.collections
    finally:
        plt.close(fig)


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("resolution", [None, 10])
@pytest.mark.parametrize("reverse", [False, True])
def test_hdf5_contributions(full, resolution, reverse, tmp_path):
    """Simple/full selection preserves component values and rebinning results."""
    from taurex.plot.plotter import Plotter

    grid = np.array([0.5, 2.0, 7.0])
    if reverse:
        grid = grid[::-1]
    native_grid = np.linspace(1000.0, 25000.0, 1000)
    fig, ax = plt.subplots()
    try:
        with h5py.File(tmp_path / "contributions.h5", "w") as handle:
            handle["native_wngrid"] = native_grid
            contributions = handle.create_group("Contributions")
            absorption = contributions.create_group("Absorption")
            rayleigh = contributions.create_group("Rayleigh")
            water = absorption.create_group("H2O")
            methane = absorption.create_group("CH4")
            hydrogen = rayleigh.create_group("H2")
            for idx, group in enumerate((water, methane, rayleigh, hydrogen), start=1):
                group["binned_spectrum"] = np.array([idx * 0.001, 0.0, idx * 0.002])
                group["binned_std"] = np.full(3, idx * 0.0001)
                group["native_spectrum"] = idx * (
                    0.001 + 0.0001 * np.sin(native_grid / 1000)
                )
                group["native_std"] = np.full(1000, idx * 0.0001)
            plotter = Plotter.__new__(Plotter)
            selected = {"Absorption-CH4": methane, "Absorption-H2O": water}
            selected["Rayleigh-H2" if full else "Rayleigh"] = (
                hydrogen if full else rayleigh
            )
            expected = {
                name: plotter._prepare_spectrum(grid, native_grid, group, resolution)
                for name, group in selected.items()
            }
            method = plotter.full_contrib_plot if full else plotter.simple_contrib_plot
            result = method(handle, grid, resolution=resolution)
        np.testing.assert_array_equal(result, grid)
        assert [line.get_label() for line in ax.lines] == list(selected)
        assert len(ax.collections) == 6
        for idx, (line, (wl, values, errors)) in enumerate(
            zip(ax.lines, expected.values(), strict=True)
        ):
            np.testing.assert_array_equal(line.get_xdata(), wl)
            np.testing.assert_allclose(line.get_ydata(), values, rtol=1e-12)
            for sigma, band in enumerate(
                ax.collections[2 * idx : 2 * idx + 2], start=1
            ):
                vertices = band.get_paths()[0].vertices
                for x, y, error in zip(wl, values, errors, strict=True):
                    bounds = vertices[vertices[:, 0] == x, 1]
                    np.testing.assert_allclose(
                        [bounds.min(), bounds.max()],
                        [y - sigma * error, y + sigma * error],
                        rtol=1e-12,
                    )
        fig.canvas.draw()
    finally:
        plt.close(fig)

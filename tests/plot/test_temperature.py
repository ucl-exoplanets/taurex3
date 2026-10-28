"""Temperature profiles from arrays and through the existing Plotter."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from taurex.plot import plot_temperature_pressure


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("uncertainty", [False, True])
def test_temperature_values(reverse, uncertainty):
    """Preserve layer coordinates and uncertainty bounds in either order."""
    temperature = np.array([1200.0, 800.0, 900.0])
    pressure = np.array([1e5, 3e3, 1.0])
    std = np.array([10.0, 20.0, 30.0])
    if reverse:
        temperature, pressure, std = temperature[::-1], pressure[::-1], std[::-1]
    fig, ax = plt.subplots()
    try:
        line, band = plot_temperature_pressure(
            ax,
            temperature,
            pressure,
            std=std if uncertainty else None,
            color="red",
            label="profile",
            linestyle="--",
        )
        np.testing.assert_array_equal(line.get_xdata(), temperature)
        np.testing.assert_array_equal(line.get_ydata(), pressure / 1e5)
        assert line.get_color() == "red"
        assert line.get_label() == "profile"
        assert line.get_linestyle() == "--"
        assert ax.get_ylim() == (1.0, 1e-5)
        assert ax.get_yscale() == "log"
        if uncertainty:
            vertices = band.get_paths()[0].vertices
            for temp, pres, error in zip(temperature, pressure / 1e5, std, strict=True):
                bounds = vertices[vertices[:, 1] == pres, 0]
                assert bounds.min() == temp - error
                assert bounds.max() == temp + error
        else:
            assert band is None
        fig.canvas.draw()
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "temperature,pressure,std",
    [([1000], [1e3], None), ([1000], [1e3], [0])],
)
def test_single_layer(temperature, pressure, std):
    """A single layer produces a usable logarithmic pressure axis."""
    fig, ax = plt.subplots()
    try:
        plot_temperature_pressure(ax, temperature, pressure, std=std)
        assert ax.yaxis_inverted()
        fig.canvas.draw()
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "temperature,pressure,std",
    [
        ([1000], [1e5, 1], None),
        ([[1000]], [1], None),
        ([np.nan], [1], None),
        ([0], [1], None),
        ([1000], [0], None),
        ([1000], [np.inf], None),
        ([1000, 900], [1, 1], None),
        ([], [], None),
        ([1000], [1], [1, 2]),
        ([1000], [1], [-1]),
        ([1000], [1], [np.nan]),
    ],
)
def test_invalid_profiles(temperature, pressure, std):
    """Reject invalid inputs before modifying the caller's axes."""
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError):
            plot_temperature_pressure(ax, temperature, pressure, std=std)
        assert not ax.lines and not ax.collections
    finally:
        plt.close(fig)


@pytest.mark.parametrize("fitted", [False, True])
def test_plotter_temperature_values(fitted, monkeypatch, tmp_path):
    """Forward and retrieval wrappers retain data and uncertainty values."""
    from taurex.plot.plotter import Plotter

    plotter = Plotter.__new__(Plotter)
    profiles = {
        "temp_profile": np.array([1200.0, 800.0, 900.0]),
        "temp_profile_std": np.array([10.0, 20.0, 30.0]),
        "pressure_profile": np.array([1e5, 3e3, 1.0]),
    }
    solution = {"Profiles": profiles}
    plotter.fd = {"Output": {"Solutions": {"solution0": solution}}}
    plotter.forward_output = lambda: solution
    plotter.cmap = plt.get_cmap("viridis")
    plotter.title = None
    plotter.prefix = "test"
    plotter.out_folder = str(tmp_path)
    saved = []
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: saved.append(plt.gcf()))
    monkeypatch.setattr(plt, "close", lambda *args: None)
    if fitted:
        plotter.plot_fitted_tp()
    else:
        plotter.plot_forward_tp()
    ax = saved[0].axes[0]
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), profiles["temp_profile"])
    np.testing.assert_array_equal(
        ax.lines[0].get_ydata(), profiles["pressure_profile"] / 1e5
    )
    assert ax.yaxis_inverted()
    assert len(ax.collections) == int(fitted)
    saved[0].canvas.draw()
    # Close after restoring the real pyplot function.
    monkeypatch.undo()
    plt.close(saved[0])

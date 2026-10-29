"""Standalone gas profiles and numerical compatibility with Plotter."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from taurex.plot import plot_gas_profile


def assert_band(band, pressure, lower, upper):
    """Check the drawn bounds at every layer."""
    vertices = band.get_paths()[0].vertices
    for pres, low, high in zip(pressure, lower, upper, strict=True):
        values = vertices[vertices[:, 1] == pres, 0]
        np.testing.assert_allclose(
            [values.min(), values.max()], [low, high], rtol=1e-12
        )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("uncertainty", [False, True])
def test_gas_values(reverse, uncertainty):
    """Preserve values, styles and legacy log-space uncertainty bounds."""
    pressure = np.array([1e5, 3e3, 1.0])
    profile = np.array([1e-4, 2e-6, 3e-8])
    std = np.array([0.0, 1e-6, 9e-8])
    # Independent reference: profile**2 / (profile + std).
    lower = np.array([1e-4, 4e-6 / 3, 7.5e-9])
    if reverse:
        pressure, profile, std, lower = (
            values[::-1] for values in (pressure, profile, std, lower)
        )
    fig, ax = plt.subplots()
    try:
        line, band = plot_gas_profile(
            ax,
            profile,
            pressure,
            std=std if uncertainty else None,
            color="red",
            linestyle="--",
            label="H2O",
        )
        np.testing.assert_array_equal(line.get_xdata(), profile)
        np.testing.assert_array_equal(line.get_ydata(), pressure / 1e5)
        assert line.get_color() == "red"
        assert line.get_label() == "H2O"
        assert line.get_linestyle() == "--"
        assert ax.get_xscale() == ax.get_yscale() == "log"
        assert ax.get_ylim() == (1.0, 1e-5)
        assert len(fig.axes) == 1
        if uncertainty:
            assert_band(band, pressure / 1e5, lower, profile + std)
            assert band.get_alpha() == 0.5
            np.testing.assert_allclose(band.get_facecolor()[0], [1, 0, 0, 0.5])
        else:
            assert band is None
        fig.canvas.draw()
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "profile,std", [([1e-5], None), ([1e-5], [0.0]), ([0.0, 1e-5], [0.0, 1e-6])]
)
def test_singleton_and_zero(profile, std):
    """Support a single layer and retain zeros without invalid log arithmetic."""
    pressure = np.geomspace(1e5, 1.0, len(profile))
    fig, ax = plt.subplots()
    try:
        with np.errstate(divide="raise", invalid="raise"):
            line, band = plot_gas_profile(ax, profile, pressure, std=std)
        np.testing.assert_array_equal(line.get_xdata(), profile)
        assert ax.yaxis_inverted()
        if band is not None:
            assert np.isfinite(band.get_paths()[0].vertices).all()
        fig.canvas.draw()
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "profile,pressure,std",
    [
        ([], [], None),
        ([[1.0]], [1.0], None),
        ([1.0], [1.0, 2.0], None),
        ([-1.0], [1.0], None),
        ([np.nan], [1.0], None),
        ([np.inf], [1.0], None),
        ([1.0], [0.0], None),
        ([1.0], [np.inf], None),
        ([1.0, 2.0], [1.0, 1.0], None),
        ([1.0, 2.0, 3.0], [1.0, 3.0, 2.0], None),
        ([1.0], [1.0], [1.0, 2.0]),
        ([1.0], [1.0], [-1.0]),
        ([1.0], [1.0], [np.nan]),
        ([1.0], [1.0], [np.inf]),
    ],
)
def test_invalid_gas(profile, pressure, std):
    """Reject invalid inputs before drawing."""
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError):
            plot_gas_profile(ax, profile, pressure, std=std)
        assert not ax.lines and not ax.collections
    finally:
        plt.close(fig)


@pytest.mark.parametrize("fitted", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_plotter_gas_values(fitted, reverse, monkeypatch, tmp_path):
    """Active and inactive gas wrappers preserve profiles and uncertainty bounds."""
    from taurex.plot.plotter import Plotter

    plotter = Plotter.__new__(Plotter)
    pressure = np.array([1e5, 3e3, 1.0])
    active = np.array([[1e-4, 2e-6, 3e-8], [1e-3, 1e-4, 1e-5]])
    inactive = np.array([[0.8, 0.7, 0.6], [0.2, 0.3, 0.4]])
    if reverse:
        pressure, active, inactive = pressure[::-1], active[:, ::-1], inactive[:, ::-1]
    profiles = {"pressure_profile": pressure}
    for name, data in (("active", active), ("inactive", inactive)):
        profiles[name + "_mix_profile"] = data
        profiles[name + "_mix_profile_std"] = data / 2
    solution = {"Profiles": profiles}
    plotter.fd = {
        "Output": {"Solutions": {"solution0": solution}},
        "ModelParameters": {
            "Chemistry": {
                "active_gases": np.array([[b"H2O"], [b"CH4"]]),
                "inactive_gases": np.array([[b"H2"], [b"He"]]),
            }
        },
    }
    plotter.forward_output = lambda: solution
    plotter.cmap = plt.get_cmap("viridis")
    plotter.title = "Test"
    plotter.prefix = "test"
    plotter.out_folder = str(tmp_path)
    saved = []
    with monkeypatch.context() as patch:
        patch.setattr(plt, "savefig", lambda *args, **kwargs: saved.append(plt.gcf()))
        patch.setattr(plt, "close", lambda *args: None)
        try:
            if fitted:
                plotter.plot_fit_xprofile()
            else:
                plotter.plot_forward_xprofile()
            assert len(saved) == 2
            for fig, data, names in zip(
                saved, (active, inactive), (["H2O", "CH4"], ["H2", "He"]), strict=True
            ):
                ax = fig.axes[0]
                assert len(ax.lines) == 2
                assert len(ax.collections) == (2 if fitted else 0)
                for idx, (line, profile, name) in enumerate(
                    zip(ax.lines, data, names, strict=True)
                ):
                    np.testing.assert_array_equal(line.get_xdata(), profile)
                    np.testing.assert_array_equal(line.get_ydata(), pressure / 1e5)
                    assert line.get_label() == name
                    if fitted:
                        # Legacy logarithmic reflection is equivalent to p / 1.5 here.
                        assert_band(
                            ax.collections[idx],
                            pressure / 1e5,
                            profile / 1.5,
                            profile * 1.5,
                        )
                assert ax.yaxis_inverted()
                assert ax.get_xlim() == (1e-12, 3)
                fig.canvas.draw()
        finally:
            patch.undo()
            for fig in saved:
                plt.close(fig)

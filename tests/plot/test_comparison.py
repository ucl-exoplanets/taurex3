"""Integration checks for combined HDF5 plots."""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest

from taurex.plot import plot_gas_profile_files
from taurex.plot import plot_posterior_files
from taurex.plot import plot_spectrum_files
from taurex.plot import plot_temperature_pressure_files


@pytest.fixture
def outputs(tmp_path):
    """Build two retrieval files with different grids and parameter orders."""
    paths = []
    rng = np.random.default_rng(42)
    samples = rng.normal(size=(200, 2))
    for index in range(2):
        path = tmp_path / f"target{index}.h5"
        paths.append(path)
        with h5py.File(path, "w") as fd:
            result = fd.create_group("Output/Solutions/solution0")
            profiles = result.create_group("Profiles")
            profiles["pressure_profile"] = np.array([1e5, 100]) / 10**index
            profiles["temp_profile"] = [1000, 1200]
            profiles["temp_profile_std"] = [10, 20]
            profiles["active_mix_profile"] = [[1e-4, 2e-4]]
            profiles["inactive_mix_profile"] = [[0.8, 0.9]]
            chemistry = fd.create_group("ModelParameters/Chemistry")
            chemistry["active_gases"] = [b"H2O"]
            chemistry["inactive_gases"] = [b"H2"]
            spectra = result.create_group("Spectra")
            spectra["binned_wlgrid"] = [1, 2]
            spectra["binned_spectrum"] = [0, -0.1]
            obs = fd.create_group("Observed")
            obs["wlgrid"] = [1, 2]
            obs["spectrum"] = [0, 0.1]
            obs["errorbars"] = [0.01, 0.02]
            fd["Optimizer/fit_parameter_names"] = (
                [b"a", b"b"] if index == 0 else [b"b", b"a"]
            )
            result["tracedata"] = samples if index == 0 else samples[:, ::-1]
            result["weights"] = np.arange(1, 201)
    yield paths
    plt.close("all")


def test_profiles_and_file_closure(outputs):
    """Retain values, labels and the union of pressure ranges."""
    fig, ax = plt.subplots()
    artists = plot_temperature_pressure_files(ax, outputs, labels=["A", "B"])
    assert [line.get_label() for line, _ in artists] == ["A", "B"]
    assert ax.get_ylim() == (1, 0.0001)
    np.testing.assert_array_equal(artists[1][0].get_ydata(), [0.1, 0.0001])
    assert all(band is not None for _, band in artists)
    for path in outputs:
        with h5py.File(path, "r+") as fd:
            fd.attrs["closed_by_loader"] = True


def test_gases_and_spectra(outputs):
    """Select gas species and retain nonpositive spectral samples."""
    _, ax = plt.subplots()
    gases = plot_gas_profile_files(ax, outputs, gases=["H2O"])
    assert len(gases) == 2
    assert all("H2O" in line.get_label() for line, _ in gases)
    _, ax = plt.subplots()
    models = plot_spectrum_files(ax, outputs)
    assert len(models) == 2
    for line, _ in models:
        np.testing.assert_array_equal(line.get_ydata(), [0, -0.1])
    assert len(ax.containers) == 2


def test_posterior_alignment(outputs, monkeypatch):
    """Align posterior columns and normalize each weight vector."""
    import taurex.plot.corner as module

    calls = []

    def capture(samples, **kwargs):
        calls.append((samples, kwargs))
        return kwargs["fig"] or plt.figure()

    monkeypatch.setattr(module, "corner", capture)
    plot_posterior_files(outputs, parameters=["a", "b"])
    np.testing.assert_array_equal(calls[0][0], calls[1][0])
    assert calls[0][1]["range"] == calls[1][1]["range"]
    assert calls[0][1]["weights"].sum() == pytest.approx(1)
    with pytest.raises(ValueError, match="missing"):
        plot_posterior_files(outputs, parameters=["unknown"])


def test_corner_render(outputs):
    """Render a real combined corner plot."""
    figure = plot_posterior_files(outputs, bins=10)
    figure.canvas.draw()
    assert len(figure.axes) == 4


def test_input_labels(outputs):
    """Reject empty file lists and inconsistent labels."""
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="labels"):
        plot_temperature_pressure_files(ax, outputs, labels=["only one"])
    with pytest.raises(ValueError, match="at least"):
        plot_temperature_pressure_files(ax, [])


def test_forward_and_multiple_solutions(outputs):
    """Include forward profiles and every retrieval solution."""
    with h5py.File(outputs[0], "r+") as fd:
        fd.copy("Output/Solutions/solution0/Profiles", "Output/Profiles")
        del fd["Output/Solutions"]
    with h5py.File(outputs[1], "r+") as fd:
        fd.copy("Output/Solutions/solution0", "Output/Solutions/solution2")
    _, ax = plt.subplots()
    artists = plot_temperature_pressure_files(ax, outputs)
    assert [line.get_label() for line, _ in artists] == [
        "target0",
        "target1 (solution0)",
        "target1 (solution2)",
    ]


def test_single_layer_overlay():
    """Include both pressures when overlaying single-layer profiles."""
    from taurex.plot import plot_temperature_pressure

    _, ax = plt.subplots()
    plot_temperature_pressure(ax, [1000], [1e5])
    plot_temperature_pressure(ax, [1100], [1e2])
    assert ax.get_ylim() == (1, 0.001)
    plt.close(ax.figure)


@pytest.mark.parametrize("legacy", [False, True])
def test_spectral_grid_fallbacks(outputs, legacy):
    """Keep spectral values paired with legacy or native coordinates."""
    for path in outputs:
        with h5py.File(path, "r+") as fd:
            spectra = fd["Output/Solutions/solution0/Spectra"]
            if legacy:
                spectra.move("binned_spectrum", "bin_spectrum")
                spectra.move("binned_wlgrid", "bin_wlgrid")
            else:
                spectra.move("binned_spectrum", "native_spectrum")
                del spectra["binned_wlgrid"]
                spectra["native_wngrid"] = [10000, 5000]
    _, ax = plt.subplots()
    artists = plot_spectrum_files(ax, outputs, observed=False)
    for line, _ in artists:
        np.testing.assert_array_equal(line.get_xdata(), [1, 2])
    assert not ax.containers


@pytest.mark.parametrize("bad", [-1, np.nan, 0])
def test_invalid_posterior_weights(outputs, bad):
    """Reject negative, nonfinite and all-zero posterior weights."""
    with h5py.File(outputs[0], "r+") as fd:
        del fd["Output/Solutions/solution0/weights"]
        fd["Output/Solutions/solution0/weights"] = np.full(200, bad)
    with pytest.raises(ValueError, match="weights"):
        plot_posterior_files(outputs)


def test_cli_combined_pdfs(outputs, tmp_path, monkeypatch):
    """Exercise argparse and render all four comparison PDFs."""
    from taurex.plot.plotter import main

    destination = tmp_path / "plots"
    monkeypatch.setattr(
        "sys.argv",
        [
            "taurex-plot",
            "-i",
            *map(str, outputs),
            "-o",
            str(destination),
            "--all",
            "--labels",
            "A",
            "B",
            "--parameters",
            "b",
            "a",
            "--prefix",
            "comparison",
            "--title",
            "Two targets",
            "-m",
            "viridis",
        ],
    )
    main()
    assert {path.name for path in destination.iterdir()} == {
        f"comparison_{suffix}.pdf"
        for suffix in ("tp_profile", "mixratio", "spectrum", "posteriors")
    }
    assert all(path.read_bytes().startswith(b"%PDF") for path in destination.iterdir())
    assert not plt.get_fignums()


@pytest.mark.parametrize(
    "extra",
    [
        ["--labels", "one"],
        ["--resolution", "100"],
        ["--plot-tau"],
        ["--plot-contrib"],
        ["--full-contrib"],
        ["--plot-cprofile"],
        ["--parameters", "a"],
    ],
)
def test_cli_rejects_unsupported_options(outputs, tmp_path, monkeypatch, extra):
    """Reject unsupported comparison arguments before producing any files."""
    from taurex.plot.plotter import main

    destination = tmp_path / "plots"
    monkeypatch.setattr(
        "sys.argv",
        [
            "taurex-plot",
            "-i",
            *map(str, outputs),
            "-o",
            str(destination),
            "--plot-tpprofile",
            *extra,
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not destination.exists()


def test_cli_single_input_legacy(outputs, tmp_path, monkeypatch):
    """Pass a scalar filename to the unchanged single-file Plotter."""
    from taurex.plot import plotter

    calls = []
    monkeypatch.setattr(
        plotter, "Plotter", lambda filename, **kwargs: calls.append(filename)
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "taurex-plot",
            "-i",
            str(outputs[0]),
            "-o",
            str(tmp_path),
        ],
    )
    plotter.main()
    assert calls == [str(outputs[0])]


def test_cli_all_requires_retrievals(outputs, tmp_path, monkeypatch):
    """Explain that --all includes posteriors and requires retrievals."""
    from taurex.plot.plotter import main

    with h5py.File(outputs[0], "r+") as fd:
        fd.copy("Output/Solutions/solution0/Profiles", "Output/Profiles")
        del fd["Output/Solutions"]
    monkeypatch.setattr(
        "sys.argv",
        [
            "taurex-plot",
            "-i",
            *map(str, outputs),
            "-o",
            str(tmp_path / "plots"),
            "--all",
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not (tmp_path / "plots").exists()

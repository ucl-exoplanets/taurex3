"""Compare TauREx HDF5 outputs on caller-owned axes.

Files are opened only while loading; returned artists never own file handles.
"""

from pathlib import Path

import h5py
import numpy as np

from .profile import plot_gas_profile
from .profile import plot_temperature_pressure
from .spectrum import plot_spectrum


def _results(filenames, labels):
    filenames = list(filenames)
    if not filenames:
        raise ValueError("at least one filename is required")
    if labels is None:
        labels = [Path(name).stem for name in filenames]
    else:
        labels = list(labels)
    if len(labels) != len(filenames):
        raise ValueError("labels must match filenames")
    for filename, label in zip(filenames, labels, strict=True):
        with h5py.File(filename, "r") as fd:
            output = fd["Output"]
            if "Solutions" in output:
                solutions = output["Solutions"]
                keys = sorted(
                    (key for key in solutions if key.startswith("solution")),
                    key=lambda key: int(key[8:]),
                )
                for key in keys:
                    name = f"{label} ({key})" if len(keys) > 1 else label
                    yield fd, solutions[key], name
            else:
                yield fd, output, label


def plot_temperature_pressure_files(ax, filenames, labels=None, **kwargs):
    """Overlay all forward/retrieval TP profiles from a sequence of paths.

    Labels default to file stems, extended with solution names for multimodal
    retrievals. Optional uncertainties are read when present. Keyword arguments
    style each line. Return a list of (line, band) pairs; no legend is created.
    """
    artists = []
    for _, result, label in _results(filenames, labels):
        profiles = result["Profiles"]
        artists.append(
            plot_temperature_pressure(
                ax,
                profiles["temp_profile"][...],
                profiles["pressure_profile"][...],
                std=_optional(profiles, "temp_profile_std"),
                label=label,
                **kwargs,
            )
        )
    return artists


def _optional(group, key):
    return group[key][...] if key in group else None


def _strings(dataset):
    return list(dataset.asstr()[...])


def plot_gas_profile_files(ax, filenames, labels=None, gases=None, **kwargs):
    """Overlay active and inactive gas profiles from multiple output files.

    ``gases`` optionally selects species by name. Each line label identifies
    its file, solution and gas. Return a list of (line, band) pairs.
    """
    artists = []
    selected = None if gases is None else set(gases)
    for fd, result, label in _results(filenames, labels):
        chemistry = fd["ModelParameters/Chemistry"]
        profiles = result["Profiles"]
        for kind in ("active", "inactive"):
            names = _strings(chemistry[f"{kind}_gases"])
            values = profiles[f"{kind}_mix_profile"][...]
            std = _optional(profiles, f"{kind}_mix_profile_std")
            for index, name in enumerate(names):
                if selected is None or name in selected:
                    artists.append(
                        plot_gas_profile(
                            ax,
                            values[index],
                            profiles["pressure_profile"][...],
                            std=None if std is None else std[index],
                            label=f"{label}: {name}",
                            **kwargs,
                        )
                    )
    return artists


def plot_spectrum_files(ax, filenames, labels=None, observed=True, **kwargs):
    """Overlay model spectra and optionally observations from output files.

    Prefer binned spectra, accepting legacy ``bin_*`` names, otherwise use
    native spectra and their wavelength grid. Retain zero and negative values.
    No rebinning is performed. Return model (line, bands) pairs. Observational
    error bars use the corresponding model color and appear once per file.
    """
    artists = []
    seen = set()
    for fd, result, label in _results(filenames, labels):
        spectra = result["Spectra"]
        if "binned_spectrum" in spectra:
            key, grid, std = "binned_spectrum", "binned_wlgrid", "binned_std"
            if grid not in spectra:
                grid = "bin_wlgrid"
        elif "bin_spectrum" in spectra:
            key, grid, std = "bin_spectrum", "bin_wlgrid", "binned_std"
        else:
            key, grid, std = "native_spectrum", "native_wlgrid", "native_std"
        wavelengths = (
            spectra[grid][...]
            if key != "native_spectrum" or grid in spectra
            else 10000 / spectra["native_wngrid"][...]
        )
        line, bands = plot_spectrum(
            ax,
            wavelengths,
            spectra[key][...],
            std=_optional(spectra, std),
            label=label,
            **kwargs,
        )
        artists.append((line, bands))
        if observed and "Observed" in fd and fd.filename not in seen:
            obs = fd["Observed"]
            ax.errorbar(
                obs["wlgrid"][...],
                obs["spectrum"][...],
                yerr=obs["errorbars"][...],
                fmt="o",
                ls="none",
                color=line.get_color(),
                label=f"{label}: observed",
            )
            seen.add(fd.filename)
    return artists


def plot_posterior_files(filenames, parameters=None, labels=None, **kwargs):
    """Overlay weighted retrieval posteriors in one corner figure.

    Parameters are aligned by name, never by column position. By default all
    parameters of the first retrieval are required in every file. An explicit
    nonempty sequence selects and orders shared fitting parameters. Derived
    parameters are excluded. Ranges cover every selected solution. Return the
    figure; file/solution labels are shown in a figure legend. Additional
    keyword arguments are passed to the bundled corner implementation.
    """
    from matplotlib.lines import Line2D

    from .corner import corner

    records = []
    for fd, result, label in _results(filenames, labels):
        names = _strings(fd["Optimizer/fit_parameter_names"])
        if parameters is None:
            parameters = names
        parameters = list(parameters)
        if not parameters or len(set(parameters)) != len(parameters):
            raise ValueError("parameters must be nonempty and unique")
        if any(name not in names for name in parameters):
            raise ValueError(f"missing selected fitting parameters in {fd.filename}")
        samples = result["tracedata"][...][:, [names.index(p) for p in parameters]]
        weights = result["weights"][...].reshape(-1)
        if (
            not np.all(np.isfinite(samples))
            or weights.shape != samples.shape[:1]
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or weights.sum() <= 0
        ):
            raise ValueError("posterior samples and weights must be finite and valid")
        records.append((samples, weights / weights.sum(), label))
    combined = np.concatenate([samples for samples, _, _ in records])
    limits = list(zip(combined.min(axis=0), combined.max(axis=0), strict=True))
    if any(low == high for low, high in limits):
        raise ValueError("selected posterior parameters must have nonzero ranges")
    kwargs.setdefault("range", limits)
    kwargs.setdefault("labels", parameters)
    fig = None
    handles = []
    for index, (samples, weights, label) in enumerate(records):
        options = dict(kwargs)
        color = options.pop("color", f"C{index % 10}")
        fig = corner(samples, weights=weights, fig=fig, color=color, **options)
        handles.append(Line2D([], [], color=color, label=label))
    fig.legend(handles=handles)
    return fig

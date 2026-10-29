"""Plot atmospheric depth profiles from arrays on caller-owned axes."""

import numpy as np


def _depth_arrays(tau, pressure):
    tau = np.asarray(tau)
    pressure = np.asarray(pressure, dtype=float)
    if pressure.ndim != 1 or pressure.size == 0:
        raise ValueError("pressure must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(pressure)) or np.any(pressure <= 0):
        raise ValueError("pressure must contain finite positive values")
    if tau.ndim != 2 or tau.shape[0] != pressure.size or tau.shape[1] == 0:
        raise ValueError(
            "tau must have shape (number of pressures, number of wavelengths)"
        )
    if pressure.size > 1 and not (
        np.all(np.diff(pressure) > 0) or np.all(np.diff(pressure) < 0)
    ):
        raise ValueError("pressure must be strictly monotonic")
    return tau, pressure / 1.0e5


def _pressure_axis(ax, pressure):
    ax.set_yscale("log")
    if pressure.size > 1:
        ax.set_ylim(pressure.max(), pressure.min())
    elif not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_ylabel("Pressure [bar]")


def _log_edges(values):
    centers = np.log10(values)
    if centers.size == 1:
        return 10.0 ** (centers[0] + np.array([-0.5, 0.5]))
    midpoints = (centers[1:] + centers[:-1]) / 2
    return 10.0 ** np.concatenate(
        (
            [2 * centers[0] - midpoints[0]],
            midpoints,
            [2 * centers[-1] - midpoints[-1]],
        )
    )


def plot_optical_depth(ax, tau, pressure, wavelengths, **kwargs):
    r"""Plot optical depth and return a Matplotlib ``QuadMesh``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to draw. No figure or colorbar is created.
    tau : array_like
        Depth values with shape ``(n_layers, n_wavelengths)``.
    pressure : array_like
        Positive, strictly monotonic layer pressures in Pa, displayed in bar.
    wavelengths : array_like
        Positive, strictly monotonic wavelengths in micrometres.
        Ascending and descending grids are both supported.
    \*\*kwargs : dict
        Passed to ``ax.pcolormesh``, for example ``cmap``, ``vmin``,
        ``vmax`` or ``norm``. Cell edges are inferred in logarithmic space;
        supported shading modes are ``"auto"`` and ``"flat"``.

    Notes
    -----
    Both axes use logarithmic scales and the pressure increases downwards.
    Set titles and limits using the supplied axes after plotting.
    """
    tau, pressure = _depth_arrays(tau, pressure)
    wavelengths = np.asarray(wavelengths, dtype=float)
    if wavelengths.ndim != 1 or wavelengths.size != tau.shape[1]:
        raise ValueError("wavelengths must match the second dimension of tau")
    if not np.all(np.isfinite(wavelengths)) or np.any(wavelengths <= 0):
        raise ValueError("wavelengths must contain finite positive values")
    if wavelengths.size > 1 and not (
        np.all(np.diff(wavelengths) > 0) or np.all(np.diff(wavelengths) < 0)
    ):
        raise ValueError("wavelengths must be strictly monotonic")
    # Interpolate cell edges in log space so logarithmic grids retain positive edges.
    kwargs.setdefault("shading", "auto")
    mesh = ax.pcolormesh(_log_edges(wavelengths), _log_edges(pressure), tau, **kwargs)
    ax.set_xscale("log")
    if wavelengths.size > 1:
        ax.set_xlim(wavelengths.min(), wavelengths.max())
    ax.set_xlabel(r"Wavelength [$\mu$m]")
    _pressure_axis(ax, pressure)
    return mesh


def plot_depth_contrib(ax, tau, pressure, **kwargs):
    """Plot mean depth over wavelength and return a Matplotlib ``Line2D``.

    ``tau`` has shape ``(n_layers, n_wavelengths)`` and ``pressure`` contains
    positive, strictly monotonic layer pressures in Pa. The arithmetic mean
    over wavelengths matches the historical TauREx Plotter convention.
    Pressure is displayed in bar on a logarithmic axis, increasing downwards.
    Additional keyword arguments are passed to ``ax.plot``.
    """
    tau, pressure = _depth_arrays(tau, pressure)
    (line,) = ax.plot(np.mean(tau, axis=1), pressure, **kwargs)
    ax.set_xlabel("Contribution")
    _pressure_axis(ax, pressure)
    return line

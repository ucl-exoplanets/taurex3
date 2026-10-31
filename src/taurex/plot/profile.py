"""Plot atmospheric profiles from arrays on caller-owned axes."""

import numpy as np


def _depth_arrays(tau, pressure):
    tau = np.asarray(tau)
    pressure = _pressure_array(pressure)
    if tau.ndim != 2 or tau.shape[0] != pressure.size or tau.shape[1] == 0:
        raise ValueError(
            "tau must have shape (number of pressures, number of wavelengths)"
        )
    return tau, pressure


def _pressure_array(pressure):
    pressure = np.asarray(pressure, dtype=float)
    if pressure.ndim != 1 or pressure.size == 0:
        raise ValueError("pressure must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(pressure)) or np.any(pressure <= 0):
        raise ValueError("pressure must contain finite positive values")
    if pressure.size > 1 and not (
        np.all(np.diff(pressure) > 0) or np.all(np.diff(pressure) < 0)
    ):
        raise ValueError("pressure must be strictly monotonic")
    return pressure / 1.0e5


def plot_temperature_pressure(ax, temperature, pressure, std=None, **kwargs):
    r"""Plot a temperature–pressure profile from arrays.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Caller-owned axes. No figure is created or saved.
    temperature : array_like
        One-dimensional, finite positive temperatures in K.
    pressure : array_like
        Matching positive, finite, strictly monotonic pressures in Pa.
        Ascending and descending grids are supported; displayed in bar.
    std : array_like, optional
        Matching finite nonnegative temperature standard deviations in K.
        Drawn as a band at temperature plus/minus std, with alpha 0.5
        and the same color as the line.
    \*\*kwargs : dict
        Passed to ``ax.plot``, including color, label and linestyle.

    Returns
    -------
    line : matplotlib.lines.Line2D
        The temperature profile.
    band : matplotlib.collections.FillBetweenPolyCollection or None
        The uncertainty band, or None when std is omitted.

    Notes
    -----
    Pressure uses a logarithmic axis increasing downwards. Set titles and
    limits on the supplied axes after plotting. No legend is created.
    """
    pressure = _pressure_array(pressure)
    temperature = np.asarray(temperature, dtype=float)
    if temperature.shape != pressure.shape:
        raise ValueError("temperature must be one-dimensional and match pressure")
    if not np.all(np.isfinite(temperature)) or np.any(temperature <= 0):
        raise ValueError("temperature must contain finite positive values")
    if std is not None:
        std = np.asarray(std, dtype=float)
        if std.shape != temperature.shape:
            raise ValueError("std must match temperature")
        if not np.all(np.isfinite(std)) or np.any(std < 0):
            raise ValueError("std must contain finite nonnegative values")
    (line,) = ax.plot(temperature, pressure, **kwargs)
    band = None
    if std is not None:
        band = ax.fill_betweenx(
            pressure,
            temperature - std,
            temperature + std,
            color=line.get_color(),
            alpha=0.5,
        )
    ax.set_xlabel("Temperature (K)")
    _pressure_axis(ax, pressure)
    return line, band


def plot_gas_profile(ax, mixing_ratio, pressure, std=None, **kwargs):
    r"""Plot one gas mixing-ratio profile and an optional uncertainty band.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Caller-owned axes. No figure, legend or file is created.
    mixing_ratio : array_like
        One-dimensional finite nonnegative volume mixing ratios, one per layer.
        Zero values are retained but cannot be displayed on a logarithmic axis.
    pressure : array_like
        Matching finite positive pressures in Pa, displayed in bar.
        Must be strictly monotonic in either direction; a single layer is allowed.
    std : array_like, optional
        Matching finite nonnegative standard deviations in mixing ratio.
    \*\*kwargs : dict
        Passed to ``ax.plot``, for example color, label or linestyle.

    Returns
    -------
    line : matplotlib.lines.Line2D
        The gas profile.
    band : matplotlib.collections.FillBetweenPolyCollection or None
        Uncertainty band, or None if std is omitted.

    Notes
    -----
    Both axes are logarithmic, with pressure increasing downwards. The band
    uses the historical Plotter convention: upper = mixing_ratio + std and
    lower = 10**(2*log10(mixing_ratio) - log10(upper)). This is symmetric in
    log space, rather than an arithmetic plus/minus interval. At zero mixing
    ratio the lower bound is zero (the continuous limit). The band uses the
    line color and alpha 0.5. Set axis limits and titles on the supplied axes.
    """
    pressure = _pressure_array(pressure)
    mixing_ratio = np.asarray(mixing_ratio, dtype=float)
    if mixing_ratio.shape != pressure.shape:
        raise ValueError("mixing_ratio must be one-dimensional and match pressure")
    if not np.all(np.isfinite(mixing_ratio)) or np.any(mixing_ratio < 0):
        raise ValueError("mixing_ratio must contain finite nonnegative values")
    if std is not None:
        std = np.asarray(std, dtype=float)
        if std.shape != mixing_ratio.shape:
            raise ValueError("std must match mixing_ratio")
        if not np.all(np.isfinite(std)) or np.any(std < 0):
            raise ValueError("std must contain finite nonnegative values")
        upper = mixing_ratio + std
        lower = np.zeros_like(mixing_ratio)
        positive = mixing_ratio > 0
        log_profile = np.log10(mixing_ratio[positive])
        lower[positive] = 10 ** (
            log_profile - (np.log10(upper[positive]) - log_profile)
        )
    ax.set_xscale("log")
    (line,) = ax.plot(mixing_ratio, pressure, **kwargs)
    band = None
    if std is not None:
        band = ax.fill_betweenx(
            pressure, lower, upper, color=line.get_color(), alpha=0.5
        )
    ax.set_xlabel("Mixing ratio")
    _pressure_axis(ax, pressure)
    return line, band


def _pressure_axis(ax, pressure):
    ax.set_yscale("log")
    existing = [np.asarray(line.get_ydata()) for line in ax.lines]
    values = np.concatenate([pressure, *existing])
    values = values[np.isfinite(values) & (values > 0)]
    if values.max() > values.min():
        ax.set_ylim(values.max(), values.min())
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

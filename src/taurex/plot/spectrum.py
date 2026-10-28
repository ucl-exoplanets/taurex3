"""Plot spectra from arrays on caller-owned axes."""

import numpy as np


def plot_spectrum(ax, wavelengths, spectrum, std=None, **kwargs):
    r"""Plot a spectrum with optional one- and two-sigma uncertainty bands.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Caller-owned axes. No figure, legend or file is created.
    wavelengths : array_like
        Nonempty one-dimensional finite positive wavelengths in micrometres.
        The supplied order is retained; no sorting or rebinning is performed.
    spectrum : array_like
        Matching finite spectral values, including zero and negative values.
    std : array_like, optional
        Matching finite nonnegative standard deviations, in spectrum units.
    \*\*kwargs : dict
        Passed to ``ax.plot``, including color, label, alpha and linestyle.

    Returns
    -------
    line : matplotlib.lines.Line2D
        The spectrum line.
    bands : tuple
        The one- and two-sigma Matplotlib fill collections, or an empty tuple
        when std is omitted. Bands use the line color, alpha 0.5 and 0.2,
        and zorder -2 and -3 respectively, as in the historical Plotter.

    Notes
    -----
    Values and uncertainties are plotted without filtering or conversion.
    Only the wavelength label is set. Set scales, limits, the spectrum's
    ordinate label and titles on the supplied axes. No global settings change.
    """
    wavelengths = np.asarray(wavelengths, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    if wavelengths.ndim != 1 or wavelengths.size == 0:
        raise ValueError("wavelengths must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(wavelengths)) or np.any(wavelengths <= 0):
        raise ValueError("wavelengths must contain finite positive values")
    if spectrum.shape != wavelengths.shape:
        raise ValueError("spectrum must be one-dimensional and match wavelengths")
    if not np.all(np.isfinite(spectrum)):
        raise ValueError("spectrum must contain finite values")
    if std is not None:
        std = np.asarray(std, dtype=float)
        if std.shape != spectrum.shape:
            raise ValueError("std must match spectrum")
        if not np.all(np.isfinite(std)) or np.any(std < 0):
            raise ValueError("std must contain finite nonnegative values")
    (line,) = ax.plot(wavelengths, spectrum, **kwargs)
    bands = ()
    if std is not None:
        bands = tuple(
            ax.fill_between(
                wavelengths,
                spectrum - sigma * std,
                spectrum + sigma * std,
                color=line.get_color(),
                alpha=alpha,
                zorder=zorder,
                edgecolor="none",
            )
            for sigma, alpha, zorder in ((1, 0.5, -2), (2, 0.2, -3))
        )
    ax.set_xlabel(r"Wavelength ($\mu$m)")
    return line, bands

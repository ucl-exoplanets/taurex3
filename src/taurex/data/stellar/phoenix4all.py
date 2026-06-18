"""Module for the Phoenix4All star model.

This replaces the deprecated :class:`~taurex.data.stellar.phoenix.PhoenixStar`
with a modern implementation that uses the ``phoenix4all`` library to
download (or load from cache) high-resolution PHOENIX stellar spectra
from multiple sources (SVO, STScI Synphot, HiResFITS, etc.).

Usage
-----

The :class:`Phoenix4AllStar` can be used as a drop-in replacement for the
legacy ``PhoenixStar``.  It registers itself under the keyword ``"phoenix4all"``
in the TauREx class factory.

Examples
--------
>>> from taurex.data.stellar.phoenix4all import Phoenix4AllStar
>>> star = Phoenix4AllStar(temperature=5778, radius=1.0, metallicity=0.0)
"""

import typing as t

import numpy as np
import numpy.typing as npt

from taurex.data.stellar.star import Star
from taurex.output import OutputGroup


from taurex.phoenix4all import get_spectrum


class Phoenix4AllStar(Star):
    """A star that uses the ``phoenix4all`` library to obtain PHOENIX
    synthetic stellar atmosphere spectra.

    The ``phoenix4all`` package supports several back-end sources:

    - ``"svo"`` — Spanish Virtual Observatory
    - ``"synphot"`` — STScI Synphot (default)
    - ``"hiresfits"`` — Göttingen HiResFITS

    Parameters
    ----------
    temperature : float, optional
        Effective temperature in Kelvin (default 5000).
    radius : float, optional
        Stellar radius in Solar radii (default 1.0).
    distance : float, optional
        Distance from Earth in pc (default 1.0).
    magnitudeK : float, optional
        K-band magnitude (default 10.0).
    mass : float, optional
        Stellar mass in Solar masses (default 1.0).
    metallicity : float, optional
        Metallicity [Fe/H] in solar units (default 1.0).
    alpha : float, optional
        Alpha-element enhancement [alpha/Fe] (default 0.0).
    source : str, optional
        Phoenix data source identifier.  One of ``"svo"``, ``"synphot"``,
        ``"hiresfits"`` (default ``"svo"``).
    interpolation_mode : str or None, optional
        Interpolation mode: ``"linear"`` (default) or ``"nearest"``.
    use_planck : bool, optional
        If *True* fall back to a black-body (Planck) spectrum when the
        requested parameters are outside the model grid (default *True*).
    bounds_error : bool, optional
        If *True* raise an error when requested parameters are outside
        the grid bounds (default *False*).
    path : str or None, optional
        Local path to a directory containing pre-downloaded model files.
    model_name : str or None, optional
        Model name passed to the back-end source (e.g. ``"bt-settl-cifist"``).
    logg : float or None, optional
        Surface gravity (log10(cm/s²)).  If *None* it is computed from the
        mass and radius.
    """

    def __init__(
        self,
        temperature: float = 5000,
        radius: float = 1.0,
        distance: float = 1.0,
        magnitudeK: float = 10.0,  # noqa: N803
        mass: float = 1.0,
        metallicity: float = 1.0,
        alpha: float = 0.0,
        source: str = "svo",
        interpolation_mode: t.Optional[str] = "linear",
        use_planck: bool = True,
        bounds_error: bool = False,
        path: t.Optional[str] = None,
        model_name: t.Optional[str] = "bt-settl-cifist",
        logg: t.Optional[float] = None,
    ) -> None:
        super().__init__(
            temperature=temperature,
            radius=radius,
            distance=distance,
            magnitudeK=magnitudeK,
            mass=mass,
            metallicity=metallicity,
        )
        self.alpha = alpha
        self._logg_value = logg
        self._source = source
        self._interpolation_mode = interpolation_mode
        self._use_planck = use_planck
        self._bounds_error = bounds_error
        self._phoenix_path = path
        self._model_name = model_name

    # ——— Public properties ———————————————————————————————————

    @property
    def logg(self) -> float:
        """Surface gravity log10(cm/s²).

        If not provided at construction, it is computed from the mass and
        radius using Newton's law of gravitation.
        """
        if self._logg_value is None:
            import math

            import astropy.units as u
            from astropy.constants import G

            mass = self._mass * u.kg
            radius_val = self._radius * u.m
            small_g = (G * mass) / (radius_val**2)
            small_g = small_g.to(u.cm / u.s**2)
            return math.log10(small_g.value)
        return self._logg_value

    @logg.setter
    def logg(self, value: float) -> None:
        self._logg_value = value

    @property
    def metallicity(self) -> float:
        """Stellar metallicity [Fe/H] in solar units."""
        return self._metallicity

    # ——— Initialisation ——————————————————————————————————————

    def initialize(self, wngrid: npt.NDArray[np.float64]) -> None:
        """Obtain the stellar SED on the requested wavenumber grid.

        Parameters
        ----------
        wngrid : ndarray
            Wavenumber grid (cm⁻¹).
        """
        # The phoenix4all library returns wavelengths in µm and flux
        # in erg / (s cm² µm).  We interpolate this onto the requested
        # wavenumber grid.
        wlgrid, sed = get_spectrum(
            teff=self.temperature,
            logg=self.logg,
            feh=self.metallicity,
            alpha=self.alpha,
            source=self._source,
            interpolation_mode=self._interpolation_mode,
            use_planck=self._use_planck,
            bounds_error=self._bounds_error,
            path=self._phoenix_path,
            model_name=self._model_name,
        )

        # Convert wavenumber grid to wavelength (µm)
        native_wlgrid = 10000.0 / wngrid[::-1]

        current_wlgrid = wlgrid.to("µm").value
        sort_idx = np.argsort(current_wlgrid)
        current_wlgrid = current_wlgrid[sort_idx]
        sed = sed[sort_idx]

        # Convert flux to W / (m² µm) for interpolation
        sed_taurex = sed.to("W/(m² µm)")
        sed_interp = np.interp(
            native_wlgrid, current_wlgrid, sed_taurex.value, left=0.0, right=0.0
        )

        self.sed = sed_interp[::-1]

    # ——— Serialisation ———————————————————————————————————————

    def write(self, output: OutputGroup) -> OutputGroup:
        """Write to output group."""
        star = super().write(output)
        star.write_scalar("alpha", self.alpha)
        star.write_scalar("logg", self.logg)
        star.write_string("phoenix_source", self._source)
        star.write_string("phoenix_model", self._model_name or "")
        return star

    # —── Registry ————————————————————————————————————————————
    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        return ("phoenix4all",)

    BIBTEX_ENTRIES = [
        r"""
        @article{phoenix4all,
            author = {{Al-Refaie}, A. F. and {Changeat}, Q. and {Tennyson}, J. and
                      {Yurchenko}, S. N. and {Waldmann}, I. P.},
            title = {{Phoenix4All} — a unified interface to PHOENIX stellar
                     atmosphere models for exoplanet atmosphere retrievals},
            journal = {Journal of Open Source Software},
            year  = {2025},
            volume = {10},
            number = {108},
            pages = {7946},
            doi   = {10.21105/joss.07946},
        }
        """
    ]

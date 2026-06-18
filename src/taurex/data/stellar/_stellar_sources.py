"""PHOENIX spectral source backends — core base, registry, and all sources.

This module provides the machinery to fetch, cache, and interpolate
PHOENIX stellar atmosphere spectra from multiple remote archives.
It is used internally by :class:`~taurex.data.stellar.phoenix4all.Phoenix4AllStar`.
"""

import abc
import bisect
import enum
import io
import pathlib
import re
from typing import Optional
from typing import Sequence
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from astropy import constants as const
from astropy import units as u
from astropy.io import fits
from bs4 import BeautifulSoup
from scipy.interpolate import RegularGridInterpolator

from ._stellar_io import download_to_directory
from ._stellar_io import is_remote_url
from ._stellar_io import json_unzip
from ._stellar_io import parse_directory_listing


# ======================================================================
#  Planck helper
# ======================================================================

_HC = const.h * const.c
_HC_OVER_K = (const.h * const.c / const.k_B).to("K um").value


def _planck_spectrum(wavelength: u.Quantity, temperature: u.Quantity) -> u.Quantity:
    """Black-body intensity B_lambda(T)."""
    lam = wavelength.to("um", equivalencies=u.spectral()).value
    T = temperature.to("K").value
    prefac = 2.0 * _HC.to("erg um").value
    with np.errstate(over="ignore"):
        arg = _HC_OVER_K / (lam * T)
        arg = np.clip(arg, None, 700)
        bb = prefac / lam**5 / (np.exp(arg) - 1.0)
    return (bb * u.erg / (u.s * u.cm**2 * u.um)).to(
        u.W / (u.m**2 * u.um),
        equivalencies=u.spectral_density(1 * u.um),
    )


# ======================================================================
#  Interpolation mode
# ======================================================================


class InterpolationMode(str, enum.Enum):
    """Interpolation method for the spectral grid."""

    LINEAR = "linear"
    NEAREST = "nearest"
    CUBIC = "cubic"
    QUINTIC = "quintic"


# ======================================================================
#  Phoenix data file  (one set of spectra at a single grid point)
# ======================================================================


class PhoenixDataFile:
    """Container for one PHOENIX spectrum file."""

    def __init__(
        self,
        *,
        teff: float,
        logg: float,
        feh: float,
        alpha: float,
        wlen: np.ndarray,
        flux: np.ndarray,
        header: Optional[dict] = None,
    ) -> None:
        """Store a single PHOENIX spectrum and its parameters."""
        self.teff = teff
        self.logg = logg
        self.feh = feh
        self.alpha = alpha
        self._wlen = wlen
        self._flux = flux
        self.header = header or {}

    @property
    def wlen(self) -> np.ndarray:
        """Wavelength grid in microns."""
        return self._wlen

    @property
    def flux(self) -> np.ndarray:
        """Flux values."""
        return self._flux

    def __repr__(self) -> str:
        """Return a string representation of the data file."""
        return (
            f"{type(self).__name__}(teff={self.teff}, logg={self.logg}, "
            f"feh={self.feh}, alpha={self.alpha})"
        )


# ======================================================================
#  Base source
# ======================================================================


class PhoenixSource(abc.ABC):
    """Abstract base for a PHOENIX spectral source.

    Subclasses must override :meth:`download_teff`, :meth:`download_logg`,
    :meth:`download_feh`, :meth:`download_alpha`, and :meth:`_fetch_spectrum`.
    """

    def __init__(
        self,
        *,
        path: Optional[pathlib.Path] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        interpolation_mode: InterpolationMode = InterpolationMode.LINEAR,
        **kwargs,
    ) -> None:
        """Initialise the source with optional path, URL, and interpolation."""
        self._path = path
        self._base_url = base_url
        self._model_name = model_name
        self._interpolation_mode = interpolation_mode

    # -- Grid axes (abstract) --

    @abc.abstractmethod
    def download_teff(self) -> Sequence[float]:
        """Return sorted list of available Teff values."""
        ...

    @abc.abstractmethod
    def download_logg(self) -> Sequence[float]:
        """Return sorted list of available log(g) values."""
        ...

    @abc.abstractmethod
    def download_feh(self) -> Sequence[float]:
        """Return sorted list of available [Fe/H] values."""
        ...

    @abc.abstractmethod
    def download_alpha(self) -> Sequence[float]:
        """Return sorted list of available [alpha/Fe] values."""
        ...

    @abc.abstractmethod
    def _fetch_spectrum(
        self, teff: float, logg: float, feh: float, alpha: float
    ) -> PhoenixDataFile:
        """Download or load the spectrum for the given parameters."""
        ...

    # -- Cached grid accessors --

    @property
    def teff_grid(self) -> Sequence[float]:
        """Cached Teff grid."""
        return self._teffs()

    @property
    def logg_grid(self) -> Sequence[float]:
        """Cached log(g) grid."""
        return self._loggs()

    @property
    def feh_grid(self) -> Sequence[float]:
        """Cached [Fe/H] grid."""
        return self._fehs()

    @property
    def alpha_grid(self) -> Sequence[float]:
        """Cached [alpha/Fe] grid."""
        return self._alphas()

    def _teffs(self) -> Sequence[float]:
        if not hasattr(self, "_cached_teffs"):
            self._cached_teffs = self.download_teff()
        return self._cached_teffs

    def _loggs(self) -> Sequence[float]:
        if not hasattr(self, "_cached_loggs"):
            self._cached_loggs = self.download_logg()
        return self._cached_loggs

    def _fehs(self) -> Sequence[float]:
        if not hasattr(self, "_cached_fehs"):
            self._cached_fehs = self.download_feh()
        return self._cached_fehs

    def _alphas(self) -> Sequence[float]:
        if not hasattr(self, "_cached_alphas"):
            self._cached_alphas = self.download_alpha()
        return self._cached_alphas

    # -- Public API --

    def spectrum(
        self,
        teff: float,
        logg: float,
        feh: float,
        alpha: float,
        *,
        use_planck: bool = False,
        bounds_error: bool = False,
    ) -> tuple[u.Quantity, u.Quantity]:
        """Return (wavelength, flux) for the given stellar parameters."""
        self._check_bounds(teff, logg, feh, alpha, bounds_error, use_planck)
        if use_planck and self._is_outside(teff, logg, feh, alpha):
            nearest = self._nearest_spectrum(teff, logg, feh, alpha)
            wlen = nearest.wlen
            flux = _planck_spectrum(
                wavelength=wlen * u.um,
                temperature=teff * u.K,
            )
            return (wlen * u.um, flux)
        return self._interpolate(teff, logg, feh, alpha)

    # -- Internals --

    def _is_outside(self, teff, logg, feh, alpha) -> bool:
        gf = self._all_grids_or_none()
        if gf is None:
            return True
        teffs, loggs, fehs, alphas = gf
        return not (
            min(teffs) <= teff <= max(teffs)
            and min(loggs) <= logg <= max(loggs)
            and min(fehs) <= feh <= max(fehs)
            and min(alphas) <= alpha <= max(alphas)
        )

    def _nearest_spectrum(self, teff, logg, feh, alpha) -> PhoenixDataFile:
        gf = self._all_grids_or_none()
        if gf is None:
            return self._fetch_spectrum(teff, logg, feh, alpha)
        teffs, loggs, fehs, alphas = gf

        def _clamp(xs, v):
            return max(0, min(bisect.bisect_left(xs, v), len(xs) - 1))

        return self._fetch_spectrum(
            teff=teffs[_clamp(teffs, teff)],
            logg=loggs[_clamp(loggs, logg)],
            feh=fehs[_clamp(fehs, feh)],
            alpha=alphas[_clamp(alphas, alpha)],
        )

    def _check_bounds(self, teff, logg, feh, alpha, bounds_error, use_planck):
        if not bounds_error:
            return
        gf = self._all_grids_or_none()
        if gf is not None:
            teffs, loggs, fehs, alphas = gf
            if not (
                min(teffs) <= teff <= max(teffs)
                and min(loggs) <= logg <= max(loggs)
                and min(fehs) <= feh <= max(fehs)
                and min(alphas) <= alpha <= max(alphas)
            ):
                raise ValueError(
                    f"Parameters ({teff}, {logg}, {feh}, {alpha}) "
                    f"are outside the grid bounds."
                )

    def _all_grids_or_none(self):
        try:
            return (
                list(self._teffs()),
                list(self._loggs()),
                list(self._fehs()),
                list(self._alphas()),
            )
        except Exception:
            return None

    def _interpolate(self, teff, logg, feh, alpha):
        grid = self._build_interpolation_grid(teff, logg, feh, alpha)
        if grid is None:
            pdf = self._fetch_spectrum(teff, logg, feh, alpha)
            return (pdf.wlen * u.um, pdf.flux * u.W / (u.m**2 * u.um))
        teffs, loggs, fehs, alphas, spectra, wlen = grid
        interp = RegularGridInterpolator(
            (teffs, loggs, fehs, alphas),
            spectra,
            method=self._interpolation_mode.value,
        )
        flux = interp((teff, logg, feh, alpha))
        return (wlen * u.um, flux * u.W / (u.m**2 * u.um))

    def _build_interpolation_grid(self, teff, logg, feh, alpha):
        teffs = self._find_bracket(self._teffs(), teff)
        loggs = self._find_bracket(self._loggs(), logg)
        fehs = self._find_bracket(self._fehs(), feh)
        alphas = self._find_bracket(self._alphas(), alpha)
        if not all([teffs, loggs, fehs, alphas]):
            return None
        spectra = []
        wlen = None
        for t in teffs:
            for lg in loggs:
                for f in fehs:
                    for a in alphas:
                        pdf = self._fetch_spectrum(t, lg, f, a)
                        if wlen is None:
                            wlen = pdf.wlen
                        spectra.append(pdf.flux)
        spectra_arr = np.array(spectra).reshape(
            len(teffs), len(loggs), len(fehs), len(alphas), -1
        )
        return teffs, loggs, fehs, alphas, spectra_arr, wlen

    @staticmethod
    def _find_bracket(grid: Sequence[float], val: float) -> Optional[list[float]]:
        gs = sorted(grid)
        if val <= gs[0]:
            return [gs[0]]
        if val >= gs[-1]:
            return [gs[-1]]
        idx = bisect.bisect_left(gs, val)
        lo, hi = gs[idx - 1], gs[idx]
        if abs(lo - val) < 1e-6:
            return [lo]
        if abs(hi - val) < 1e-6:
            return [hi]
        return [lo, hi]


# ======================================================================
#  Registry
# ======================================================================

_source_registry: dict[str, type[PhoenixSource]] = {}


def register_source(name: str, klass: type[PhoenixSource]) -> None:
    """Register a source class under *name*."""
    _source_registry[name] = klass


def find_source(name: str) -> type[PhoenixSource]:
    """Look up a source class by name."""
    try:
        return _source_registry[name]
    except KeyError:
        raise KeyError(
            f"Unknown source {name!r}. "
            f"Available: [{', '.join(_source_registry)}]"
        ) from None


def list_sources() -> list[str]:
    """Return registered source names."""
    return list(_source_registry)


# ======================================================================
#  SVO source
# ======================================================================

_SVO_BASE = "http://svo2.cab.inta-csic.es/svo/theory/phoenix/"
_SVO_MODEL = "phoenix"
_SVO_DATASET = "svo_dataset.jsonz"


def _load_svo_dataset():
    here = pathlib.Path(__file__).resolve().parent
    cache_file = here / "_phoenix_cache" / _SVO_DATASET
    if cache_file.exists():
        return json_unzip(cache_file)
    return {}


class SVOSource(PhoenixSource):
    """PHOENIX spectra from the Spanish Virtual Observatory."""

    def __init__(
        self,
        *,
        path=None,
        base_url=None,
        model_name=None,
        interpolation_mode=InterpolationMode.LINEAR,
        **kwargs,
    ):
        """Initialise SVO source with optional dataset cache."""
        super().__init__(
            path=path,
            base_url=base_url or _SVO_BASE,
            model_name=model_name or _SVO_MODEL,
            interpolation_mode=interpolation_mode,
        )
        self._dataset = _load_svo_dataset()

    def download_teff(self) -> Sequence[float]:
        """Return sorted list of Teff values from the SVO grid."""
        data = self._dataset.get("data", {})
        teffs = list(data.get("teff", {}).keys())
        if not teffs:
            return self._scrape_teff()
        return sorted(float(t) for t in teffs)

    def download_logg(self) -> Sequence[float]:
        """Return sorted list of log(g) values from the SVO grid."""
        data = self._dataset.get("data", {})
        loggs = list(data.get("logg", {}).keys())
        if not loggs:
            return self._scrape_logg()
        return sorted(float(g) for g in loggs)

    def download_feh(self) -> Sequence[float]:
        """Return sorted list of [Fe/H] values from the SVO grid."""
        data = self._dataset.get("data", {})
        fehs = list(data.get("feh", {}).keys())
        if not fehs:
            return self._scrape_feh()
        return sorted(float(f) for f in fehs)

    def download_alpha(self) -> Sequence[float]:
        """Return sorted list of [alpha/Fe] values from the SVO grid."""
        data = self._dataset.get("data", {})
        alphas = list(data.get("alpha", {}).keys())
        if not alphas:
            return self._scrape_alpha()
        return sorted(float(a) for a in alphas)

    def _scrape_teff(self) -> Sequence[float]:
        return self._scrape_col(0)

    def _scrape_logg(self) -> Sequence[float]:
        return self._scrape_col(1)

    def _scrape_feh(self) -> Sequence[float]:
        return self._scrape_col(2)

    def _scrape_alpha(self) -> Sequence[float]:
        return self._scrape_col(3)

    def _scrape_col(self, col: int) -> list[float]:
        url = urljoin(self._base_url, "grid/")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        vals = set()
        for row in soup.select("table tr"):
            cells = row.find_all("td")
            if len(cells) >= 4:
                try:
                    vals.add(float(cells[col].get_text(strip=True)))
                except ValueError:
                    continue
        return sorted(vals)

    def _fetch_spectrum(self, teff, logg, feh, alpha) -> PhoenixDataFile:
        teff_s = f"{int(teff):05d}"
        logg_s = f"{logg:+.1f}"
        feh_s = f"{feh:+.1f}"
        filename = (
            f"lte{teff_s}-{logg_s}{feh_s}.PHOENIX-"
            f"ACES-AGSS-COND-2011-Spec{feh_s}.Alpha={alpha:+.1f}.7.gz.txt"
        )
        url = urljoin(self._base_url, f"zones/StarWars/{feh_s}/{filename}")

        if self._path and not is_remote_url(str(self._path)):
            local = pathlib.Path(self._path) / filename
            if not local.exists():
                download_to_directory(url, pathlib.Path(self._path))
            fits_path = local
        else:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            return _read_fits_spectrum(io.BytesIO(r.content), teff, logg, feh, alpha)

        return _read_fits_spectrum(fits_path, teff, logg, feh, alpha)


# ======================================================================
#  Synphot source
# ======================================================================

_SYNPHOT_BASE = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/"
_SYNPHOT_CAT = "grid-spectroscopy-flux-SYNPHOT-v1.fits"

_PHOENIX_RE = re.compile(
    r"lte(?P<teff>\d+)-(?P<logg>[+-]?\d+\.?\d*)(?P<feh>[+-]?\d+\.?\d*)?",
    re.IGNORECASE,
)


class SynphotSource(PhoenixSource):
    """PHOENIX spectra from the STScI Synphot reference atlas."""

    def __init__(
        self,
        *,
        path=None,
        base_url=None,
        model_name=None,
        interpolation_mode=InterpolationMode.LINEAR,
        **kwargs,
    ):
        """Initialise Synphot source with optional catalog path."""
        super().__init__(
            path=path,
            base_url=base_url or _SYNPHOT_BASE,
            model_name=model_name or "synphot",
            interpolation_mode=interpolation_mode,
        )
        self._catalog: Optional[pd.DataFrame] = None

    def _load_catalog(self) -> pd.DataFrame:
        if self._catalog is not None:
            return self._catalog
        url = urljoin(self._base_url, _SYNPHOT_CAT)
        with fits.open(url) as hdul:
            table = hdul[1].data
            self._catalog = pd.DataFrame(
                {
                    "filename": [r["filename"] for r in table],
                    "teff": [r["Teff"] for r in table],
                    "logg": [r["logg"] for r in table],
                    "feh": [r["Z"] for r in table],
                    "alpha": [0.0] * len(table),
                }
            )
        return self._catalog

    def download_teff(self) -> Sequence[float]:
        """Return sorted Teff values from the Synphot catalog."""
        return sorted(self._load_catalog()["teff"].unique().tolist())

    def download_logg(self) -> Sequence[float]:
        """Return sorted log(g) values from the Synphot catalog."""
        return sorted(self._load_catalog()["logg"].unique().tolist())

    def download_feh(self) -> Sequence[float]:
        """Return sorted [Fe/H] values from the Synphot catalog."""
        return sorted(self._load_catalog()["feh"].unique().tolist())

    def download_alpha(self) -> Sequence[float]:
        """Return sorted [alpha/Fe] values from the Synphot catalog."""
        return sorted(self._load_catalog()["alpha"].unique().tolist())

    def _fetch_spectrum(self, teff, logg, feh, alpha) -> PhoenixDataFile:
        cat = self._load_catalog()
        mask = (
            (np.abs(cat["teff"] - teff) < 1e-3)
            & (np.abs(cat["logg"] - logg) < 1e-3)
            & (np.abs(cat["feh"] - feh) < 1e-3)
        )
        matches = cat[mask]
        if matches.empty:
            raise FileNotFoundError(
                f"No spectrum for teff={teff}, logg={logg}, feh={feh}"
            )
        filename = matches.iloc[0]["filename"]
        url = urljoin(self._base_url, filename)
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        return _read_fits_spectrum(io.BytesIO(r.content), teff, logg, feh, alpha)


# ======================================================================
#  HiResFITS source
# ======================================================================

_HIRESFITS_BASE = "https://www2.mpia-hd.mpg.de/~ansgar/hiresfits/"
_HIRESFITS_CACHE = "hiresfit_cache.jsonz"
_HIRESFITS_RE = re.compile(
    r"lte(?P<teff>\d+)-(?P<logg>[+-]?\d+\.?\d*)(?P<feh>[+-]?\d+\.?\d*)\.", re.IGNORECASE
)


def _load_hiresfits_cache():
    here = pathlib.Path(__file__).resolve().parent
    cf = here / "_phoenix_cache" / _HIRESFITS_CACHE
    if cf.exists():
        return json_unzip(cf)
    return {}


class HiResFitsSource(PhoenixSource):
    """High-resolution PHOENIX spectra from the MPIA directory."""

    def __init__(
        self,
        *,
        path=None,
        base_url=None,
        model_name=None,
        interpolation_mode=InterpolationMode.LINEAR,
        **kwargs,
    ):
        """Initialise HiResFITS source with optional cache."""
        super().__init__(
            path=path,
            base_url=base_url or _HIRESFITS_BASE,
            model_name=model_name or "hiresfits",
            interpolation_mode=interpolation_mode,
        )
        self._listing: list[dict] = []
        self._cache = _load_hiresfits_cache()

    def _get_listing(self) -> list[dict]:
        if self._listing:
            return self._listing
        if self._cache:
            self._listing = self._cache.get("entries", [])
            return self._listing
        self._listing = parse_directory_listing(self._base_url)
        return self._listing

    def _params(self) -> list[tuple]:
        entries = self._get_listing()
        results = []
        for e in entries:
            m = _HIRESFITS_RE.search(pathlib.Path(e["href"]).name)
            if m:
                results.append(
                    (
                        float(m.group("teff")),
                        float(m.group("logg")),
                        float(m.group("feh")),
                        0.0,
                    )
                )
        return results

    def download_teff(self) -> Sequence[float]:
        """Return sorted Teff values from the HiResFITS listing."""
        return sorted({p[0] for p in self._params()})

    def download_logg(self) -> Sequence[float]:
        """Return sorted log(g) values from the HiResFITS listing."""
        return sorted({p[1] for p in self._params()})

    def download_feh(self) -> Sequence[float]:
        """Return sorted [Fe/H] values from the HiResFITS listing."""
        return sorted({p[2] for p in self._params()})

    def download_alpha(self) -> Sequence[float]:
        """Return sorted [alpha/Fe] (always [0.0] for HiResFITS)."""
        return [0.0]

    def _fetch_spectrum(self, teff, logg, feh, alpha) -> PhoenixDataFile:
        for e in self._get_listing():
            m = _HIRESFITS_RE.search(pathlib.Path(e["href"]).name)
            if (
                m
                and abs(float(m.group("teff")) - teff) < 1
                and abs(float(m.group("logg")) - logg) < 0.01
                and abs(float(m.group("feh")) - feh) < 0.01
            ):
                url = (
                    e["href"]
                    if e["href"].startswith("http")
                    else urljoin(self._base_url, e["href"])
                )
                r = requests.get(url, timeout=120)
                r.raise_for_status()
                return _read_fits_spectrum(
                    io.BytesIO(r.content), teff, logg, feh, alpha
                )
        raise FileNotFoundError(
            f"No HiResFITS file for teff={teff}, logg={logg}, feh={feh}"
        )


# ======================================================================
#  Pollux stub
# ======================================================================


class PolluxSource(PhoenixSource):
    """PHOENIX spectra from Pollux (placeholder — not implemented)."""

    def __init__(self, **kwargs):
        """Initialise Pollux source (stub — raises NotImplementedError)."""
        super().__init__(**kwargs)
        raise NotImplementedError("PolluxSource is not yet implemented")

    def download_teff(self) -> Sequence[float]:
        """Not implemented."""
        raise NotImplementedError

    def download_logg(self) -> Sequence[float]:
        """Not implemented."""
        raise NotImplementedError

    def download_feh(self) -> Sequence[float]:
        """Not implemented."""
        raise NotImplementedError

    def download_alpha(self) -> Sequence[float]:
        """Not implemented."""
        raise NotImplementedError

    def _fetch_spectrum(self, teff, logg, feh, alpha) -> PhoenixDataFile:
        """Not implemented."""
        raise NotImplementedError


# ======================================================================
#  FITS reader helper
# ======================================================================


def _read_fits_spectrum(source, teff, logg, feh, alpha) -> PhoenixDataFile:
    """Read a PHOENIX FITS file and return a PhoenixDataFile."""
    close = isinstance(source, (str, pathlib.Path))
    hdul = fits.open(source)
    try:
        data = hdul[0].data
        hdr = hdul[0].header
    finally:
        if close:
            hdul.close()

    wlen = (
        np.arange(hdr.get("NAXIS1", len(data)), dtype=float)
        + hdr.get("CRPIX1", 1.0)
        - 1.0
    ) * hdr.get("CDELT1", 0.1) + hdr.get("CRVAL1", 1000.0)

    unit = str(hdr.get("CUNIT1", "Angstrom")).strip().upper()
    if unit.startswith("A"):
        wlen = wlen / 10_000.0
    elif unit == "NM":
        wlen = wlen / 1000.0

    flux = data * 1e-9
    return PhoenixDataFile(
        teff=teff, logg=logg, feh=feh, alpha=alpha, wlen=wlen, flux=flux
    )


# ======================================================================
#  Auto-register sources
# ======================================================================

register_source("svo", SVOSource)
register_source("synphot", SynphotSource)
register_source("hiresfits", HiResFitsSource)


# ======================================================================
#  Public entry point
# ======================================================================


def get_spectrum(
    *,
    teff: float,
    logg: float,
    feh: float,
    alpha: float,
    source: str = "synphot",
    interpolation_mode: InterpolationMode = InterpolationMode.LINEAR,
    use_planck: bool = False,
    bounds_error: bool = False,
    path: Optional[pathlib.Path] = None,
    base_url: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs,
) -> tuple[u.Quantity, u.Quantity]:
    """All-in-one loader for PHOENIX spectra.

    Parameters
    ----------
    teff : float
        Effective temperature in Kelvin.
    logg : float
        Surface gravity in cgs.
    feh : float
        Metallicity [Fe/H].
    alpha : float
        Alpha enhancement [alpha/Fe].
    source : str, optional
        Source identifier (``"svo"``, ``"synphot"``, ``"hiresfits"``).
    interpolation_mode : InterpolationMode, optional
        Interpolation mode (default linear).
    use_planck : bool, optional
        Fall back to black body when outside grid bounds.
    bounds_error : bool, optional
        Raise on out-of-bounds parameters.
    path : Path or None, optional
        Local download directory.
    base_url : str or None, optional
        Override the default base URL for the source.
    model_name : str or None, optional
        Model name passed to the back-end source.

    Returns
    -------
    tuple of Quantity
        (wavelength, flux).
    """
    klass = find_source(source)
    phoenix_source = klass(
        path=path,
        base_url=base_url,
        model_name=model_name,
        interpolation_mode=interpolation_mode,
        **kwargs,
    )
    return phoenix_source.spectrum(
        teff=teff,
        logg=logg,
        feh=feh,
        alpha=alpha,
        use_planck=use_planck,
        bounds_error=bounds_error,
    )

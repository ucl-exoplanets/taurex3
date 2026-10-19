"""Tests for Phoenix4AllStar and the integrated phoenix4all backend."""

import io
from unittest.mock import patch

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits

from taurex.data.stellar._stellar_sources import InterpolationMode
from taurex.data.stellar._stellar_sources import PhoenixDataFile
from taurex.data.stellar._stellar_sources import PhoenixSource
from taurex.data.stellar._stellar_sources import _read_fits_spectrum
from taurex.data.stellar._stellar_sources import find_source
from taurex.data.stellar._stellar_sources import get_spectrum
from taurex.data.stellar._stellar_sources import list_sources
from taurex.data.stellar._stellar_sources import register_source
from taurex.data.stellar.phoenix4all import Phoenix4AllStar


RSOL = 6.957e8  # Solar radius in metres


def test_phoenix4all_input_keywords():
    """Phoenix4AllStar registers under 'phoenix4all' keyword."""
    assert "phoenix4all" in Phoenix4AllStar.input_keywords()


def test_phoenix4all_creation():
    """Can instantiate Phoenix4AllStar with default parameters."""
    star = Phoenix4AllStar(temperature=5778, radius=1.0, metallicity=0.0)
    assert star.temperature == 5778
    # Star base class converts radius to metres internally
    assert star.radius == pytest.approx(RSOL)
    assert star.metallicity == 0.0
    assert star.alpha == 0.0


def test_phoenix4all_logg_computed():
    """Surface gravity is computed from mass and radius when not given."""
    star = Phoenix4AllStar(temperature=5000, radius=1.0, mass=1.0)
    # logg ~ 4.44 for Sun-like star
    assert star.logg == pytest.approx(4.44, abs=0.1)


def test_phoenix4all_logg_custom():
    """Surface gravity can be explicitly specified."""
    star = Phoenix4AllStar(temperature=5000, radius=1.0, logg=4.5)
    assert star.logg == 4.5


@patch("taurex.data.stellar.phoenix4all.get_spectrum")
def test_phoenix4all_initialize(mock_get_spectrum):
    """initialize() calls get_spectrum and produces SED on wngrid."""
    nw = 10000
    wlgrid = np.linspace(0.3, 30.0, nw) * u.um
    flux = np.ones(nw) * 1e-9 * u.W / (u.m**2 * u.um)
    mock_get_spectrum.return_value = (wlgrid, flux)

    star = Phoenix4AllStar(temperature=5778, radius=1.0, metallicity=0.0)
    wngrid = np.linspace(300, 30000, 5000)
    star.initialize(wngrid)

    assert star.sed is not None
    assert star.sed.shape == (5000,)
    assert np.all(np.isfinite(star.sed))


@patch("taurex.data.stellar.phoenix4all.get_spectrum")
def test_phoenix4all_source_keyword(mock_get_spectrum):
    """Pass source keyword correctly."""
    nw = 1000
    wlgrid = np.linspace(0.3, 30.0, nw) * u.um
    flux = np.ones(nw) * 1e-9 * u.W / (u.m**2 * u.um)
    mock_get_spectrum.return_value = (wlgrid, flux)

    star = Phoenix4AllStar(
        temperature=5000, radius=1.0, metallicity=0.0, source="synphot"
    )
    wngrid = np.linspace(300, 30000, 1000)
    star.initialize(wngrid)

    assert star.sed is not None


# ======================================================================
#  Synthetic FITS helper
# ======================================================================


def _make_phoenix_fits(
    flux_erg: np.ndarray,
    *,
    cunit: str = "Angstrom",
    crval: float = 1000.0,
    cdelt: float = 0.1,
    crpix: float = 1.0,
) -> io.BytesIO:
    """Build an in-memory FITS file mimicking a PHOENIX spectrum."""
    hdu = fits.PrimaryHDU(data=flux_erg)
    hdu.header["CRVAL1"] = crval
    hdu.header["CDELT1"] = cdelt
    hdu.header["CRPIX1"] = crpix
    hdu.header["CUNIT1"] = cunit
    hdu.header["NAXIS1"] = len(flux_erg)
    buf = io.BytesIO()
    hdu.writeto(buf)
    buf.seek(0)
    return buf


# ======================================================================
#  _read_fits_spectrum  —  FITS parsing
# ======================================================================


class TestReadFitsSpectrum:
    """Verify that the FITS reader correctly parses PHOENIX-format files."""

    def test_angstrom_wavelength(self):
        """Wavelength in Angstrom is converted to microns."""
        npts = 1000
        flux = np.random.uniform(1e-8, 1e-6, npts)

        buf = _make_phoenix_fits(flux, cunit="Angstrom")
        result = _read_fits_spectrum(buf, 5800, 4.5, 0.0, 0.0)

        assert isinstance(result, PhoenixDataFile)
        assert result.teff == 5800
        assert result.logg == 4.5
        assert len(result.wlen) == npts
        # Angstrom → µm: / 10000
        assert result.wlen[0] == pytest.approx(1000.0 / 10000.0, rel=1e-4)
        # Flux must be scaled by 1e-9
        assert result.flux[0] == pytest.approx(flux[0] * 1e-9, rel=1e-4)

    def test_nanometer_wavelength(self):
        """Wavelength in nm is converted to microns."""
        npts = 500
        flux = np.ones(npts) * 1e-7

        buf = _make_phoenix_fits(flux, cunit="NM")
        result = _read_fits_spectrum(buf, 5000, 4.0, 0.0, 0.0)

        # nm → µm: / 1000
        assert result.wlen[0] == pytest.approx(1000.0 / 1000.0, rel=1e-4)

    def test_micron_wavelength_passthrough(self):
        """Wavelength already in microns is left unchanged."""
        npts = 200
        flux = np.ones(npts) * 1e-6

        buf = _make_phoenix_fits(flux, cunit="um")
        result = _read_fits_spectrum(buf, 4000, 5.0, 0.0, 0.0)

        # CUNIT1="um" → no unit conversion, CRVAL=1000 µm stays as-is
        assert result.wlen[0] == pytest.approx(1000.0, rel=1e-4)
        # Actually with CUNIT1="um", there's no unit conversion, it stays as-is
        assert result.wlen[0] == pytest.approx(1000.0, rel=1e-4)

    def test_flux_scaling(self):
        """Flux is always scaled by 1e-9 (erg → W/m²/µm convention)."""
        flux_in = np.array([1.0, 2.0, 3.0])
        buf = _make_phoenix_fits(flux_in, cunit="Angstrom")
        result = _read_fits_spectrum(buf, 6000, 4.0, 0.0, 0.0)
        np.testing.assert_allclose(result.flux, flux_in * 1e-9)

    def test_preserves_metadata(self):
        """Teff, logg, feh, alpha are stored correctly."""
        buf = _make_phoenix_fits(np.array([1.0]))
        result = _read_fits_spectrum(buf, 6300, 4.2, -0.3, 0.1)
        assert result.teff == 6300
        assert result.logg == 4.2
        assert result.feh == -0.3
        assert result.alpha == 0.1


# ======================================================================
#  PhoenixDataFile
# ======================================================================


class TestPhoenixDataFile:
    """Verify the PhoenixDataFile container."""

    def test_basic_container(self):
        """Container stores and returns data correctly."""
        wlen = np.array([0.3, 0.5, 0.8, 1.0])
        flux = np.array([1e-7, 2e-7, 1.5e-7, 1e-7])
        pdf = PhoenixDataFile(
            teff=5500,
            logg=4.5,
            feh=0.0,
            alpha=0.0,
            wlen=wlen,
            flux=flux,
            header={"ORIGIN": "test"},
        )

        np.testing.assert_array_equal(pdf.wlen, wlen)
        np.testing.assert_array_equal(pdf.flux, flux)
        assert pdf.teff == 5500
        assert pdf.logg == 4.5
        assert pdf.header == {"ORIGIN": "test"}

    def test_repr(self):
        """String representation includes all parameters."""
        pdf = PhoenixDataFile(
            teff=5800,
            logg=4.3,
            feh=-0.5,
            alpha=0.2,
            wlen=np.array([1.0]),
            flux=np.array([1.0]),
        )
        r = repr(pdf)
        assert "5800" in r
        assert "4.3" in r
        assert "-0.5" in r
        assert "0.2" in r


# ======================================================================
#  Source registry
# ======================================================================


class TestSourceRegistry:
    """Verify the source registry and lookup."""

    def test_default_sources_registered(self):
        """SVO, Synphot, and HiResFITS are registered by default."""
        names = list_sources()
        for expected in ("svo", "synphot", "hiresfits"):
            assert expected in names

    def test_find_known_source(self):
        """Registered sources resolve to their classes."""
        from taurex.data.stellar._stellar_sources import HiResFitsSource
        from taurex.data.stellar._stellar_sources import SVOSource
        from taurex.data.stellar._stellar_sources import SynphotSource

        assert find_source("svo") is SVOSource
        assert find_source("synphot") is SynphotSource
        assert find_source("hiresfits") is HiResFitsSource

    def test_find_unknown_raises(self):
        """Unknown source names raise KeyError."""
        with pytest.raises(KeyError, match="nonexistent"):
            find_source("nonexistent")

    def test_custom_source_registration(self):
        """Custom sources can be registered and looked up."""

        class DummySource(PhoenixSource):
            def download_teff(self):
                return [3000, 6000]

            def download_logg(self):
                return [4.0, 5.0]

            def download_feh(self):
                return [0.0]

            def download_alpha(self):
                return [0.0]

            def _fetch_spectrum(self, teff, logg, feh, alpha):
                return PhoenixDataFile(
                    teff=teff,
                    logg=logg,
                    feh=feh,
                    alpha=alpha,
                    wlen=np.array([1.0]),
                    flux=np.array([1.0]),
                )

        register_source("_dummy_test_", DummySource)
        try:
            assert find_source("_dummy_test_") is DummySource
            assert "_dummy_test_" in list_sources()
        finally:
            from taurex.data.stellar._stellar_sources import _source_registry

            _source_registry.pop("_dummy_test_", None)


# ======================================================================
#  InterpolationMode enum
# ======================================================================


class TestInterpolationMode:
    """Verify the interpolation mode enum."""

    def test_values(self):
        """Enum members have expected values."""
        assert InterpolationMode.LINEAR.value == "linear"
        assert InterpolationMode.NEAREST.value == "nearest"
        assert InterpolationMode.CUBIC.value == "cubic"
        assert InterpolationMode.QUINTIC.value == "quintic"

    def test_string_equivalence(self):
        """Enum members are also strings."""
        assert InterpolationMode.LINEAR == "linear"
        assert InterpolationMode.NEAREST == "nearest"


# ======================================================================
#  get_spectrum  —  end-to-end with a mock source
# ======================================================================


class TestGetSpectrumIntegration:
    """Integration test for get_spectrum with a mock source."""

    def test_get_spectrum_returns_quantities(self):
        """get_spectrum returns astropy Quantity tuples."""

        class MockSource(PhoenixSource):
            def download_teff(self):
                return [4000, 6000]

            def download_logg(self):
                return [4.0, 5.0]

            def download_feh(self):
                return [0.0]

            def download_alpha(self):
                return [0.0]

            def _fetch_spectrum(self, teff, logg, feh, alpha):
                return PhoenixDataFile(
                    teff=teff,
                    logg=logg,
                    feh=feh,
                    alpha=alpha,
                    wlen=np.linspace(0.3, 5.0, 1000),
                    flux=np.ones(1000) * 1e-9,
                )

        register_source("_mock_test_", MockSource)
        try:
            wl, fl = get_spectrum(
                teff=5000,
                logg=4.5,
                feh=0.0,
                alpha=0.0,
                source="_mock_test_",
            )
            assert isinstance(wl, u.Quantity)
            assert isinstance(fl, u.Quantity)
            assert wl.unit == u.um
            assert fl.unit == u.W / (u.m**2 * u.um)
            assert len(wl) == 1000
        finally:
            from taurex.data.stellar._stellar_sources import _source_registry

            _source_registry.pop("_mock_test_", None)

    def test_get_spectrum_passes_keywords(self):
        """Extra kwargs are forwarded to the source constructor."""
        captured = {}

        class KwMockSource(PhoenixSource):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                captured.update(kwargs)

            def download_teff(self):
                return [5000]

            def download_logg(self):
                return [4.5]

            def download_feh(self):
                return [0.0]

            def download_alpha(self):
                return [0.0]

            def _fetch_spectrum(self, teff, logg, feh, alpha):
                return PhoenixDataFile(
                    teff=teff,
                    logg=logg,
                    feh=feh,
                    alpha=alpha,
                    wlen=np.array([1.0]),
                    flux=np.array([1.0]),
                )

        register_source("_kw_mock_", KwMockSource)
        try:
            get_spectrum(
                teff=5000,
                logg=4.5,
                feh=0.0,
                alpha=0.0,
                source="_kw_mock_",
                path="/tmp/phoenix",  # noqa: S108
                model_name="bt-settl",
                interpolation_mode=InterpolationMode.NEAREST,
                extra_flag=True,
            )
            assert captured.get("path") == "/tmp/phoenix"  # noqa: S108
            assert captured.get("model_name") == "bt-settl"
            assert captured.get("extra_flag") is True
        finally:
            from taurex.data.stellar._stellar_sources import _source_registry

            _source_registry.pop("_kw_mock_", None)


# ======================================================================
#  Phoenix4AllStar  —  end-to-end with mock FITS source
# ======================================================================


class TestPhoenix4AllStarEndToEnd:
    """Full pipeline: mock FITS → get_spectrum → Phoenix4AllStar.initialize."""

    def _register_grid(self, pdf_list):
        """Register a mock source backed by a small grid of PhoenixDataFiles."""
        pdfs = {(p.teff, p.logg, p.feh, p.alpha): p for p in pdf_list}

        class GridSource(PhoenixSource):
            def download_teff(self):
                return sorted({t for t, _, _, _ in pdfs})

            def download_logg(self):
                return sorted({g for _, g, _, _ in pdfs})

            def download_feh(self):
                return sorted({f for _, _, f, _ in pdfs})

            def download_alpha(self):
                return sorted({a for _, _, _, a in pdfs})

            def _fetch_spectrum(self, teff, logg, feh, alpha):
                return pdfs[(teff, logg, feh, alpha)]

        register_source("_grid_", GridSource)

    def _unregister(self):
        from taurex.data.stellar._stellar_sources import _source_registry

        _source_registry.pop("_grid_", None)

    def test_full_pipeline_single_point(self):
        """Single grid point: SED is interpolated onto wngrid."""
        nwl = 500
        wlen_um = np.linspace(0.3, 5.0, nwl)
        flux_vals = np.exp(-((wlen_um - 1.0) ** 2) / 0.1)

        pdf = PhoenixDataFile(
            teff=5800,
            logg=4.5,
            feh=0.0,
            alpha=0.0,
            wlen=wlen_um,
            flux=flux_vals * 1e-9,
        )
        self._register_grid([pdf])
        try:
            star = Phoenix4AllStar(
                temperature=5800,
                radius=1.0,
                logg=4.5,
                metallicity=0.0,
                source="_grid_",
            )
            wngrid = np.linspace(300, 30000, 200)
            star.initialize(wngrid)

            assert star.sed is not None
            assert star.sed.shape == (200,)
            assert np.all(np.isfinite(star.sed))
            assert np.all(star.sed >= 0)
        finally:
            self._unregister()

    def test_sed_peak_preserved(self):
        """SED shape is preserved through interpolation."""
        nwl = 300
        wlen_um = np.linspace(0.5, 3.0, nwl)
        # Simple Gaussian peak at ~1.5 µm
        flux_vals = 1e-9 * np.exp(-((wlen_um - 1.5) ** 2) / 0.05)

        pdf = PhoenixDataFile(
            teff=5000,
            logg=4.0,
            feh=0.0,
            alpha=0.0,
            wlen=wlen_um,
            flux=flux_vals,
        )
        self._register_grid([pdf])
        try:
            star = Phoenix4AllStar(
                temperature=5000,
                radius=1.0,
                logg=4.0,
                metallicity=0.0,
                source="_grid_",
            )
            wngrid = np.linspace(500, 25000, 100)
            star.initialize(wngrid)

            # Peak should be somewhere in the middle, not at boundaries
            sed = star.sed
            peak_idx = np.argmax(sed)
            assert 10 < peak_idx < 90
        finally:
            self._unregister()

    def test_write_output_group(self):
        """write() calls super().write() and records phoenix metadata."""
        pdf = PhoenixDataFile(
            teff=5800,
            logg=4.5,
            feh=0.0,
            alpha=0.0,
            wlen=np.array([1.0]),
            flux=np.array([1e-9]),
        )
        self._register_grid([pdf])
        try:
            star = Phoenix4AllStar(
                temperature=5800,
                radius=1.0,
                logg=4.5,
                metallicity=0.0,
                source="_grid_",
                model_name="bt-settl-cifist",
            )
            from unittest.mock import MagicMock

            mock_output = MagicMock()
            mock_group = MagicMock()
            mock_output.create_group.return_value = mock_group

            result = star.write(mock_output)
            assert result is mock_group
            # Verify metadata was written
            mock_group.write_scalar.assert_any_call("alpha", 0.0)
            mock_group.write_string.assert_any_call("phoenix_source", "_grid_")
        finally:
            self._unregister()

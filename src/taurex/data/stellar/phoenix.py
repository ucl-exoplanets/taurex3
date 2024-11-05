"""Module for loading in PHOENIX spectra."""
import math
import os
import typing as t

import numpy as np
import numpy.typing as npt

from taurex.cache import GlobalCache
from taurex.constants import MSOL
from taurex.output import OutputGroup
from taurex.types import PathLike

from .star import BlackbodyStar


class PhoenixStar(BlackbodyStar):
    """A star that uses the PHOENIX synthetic stellar atmosphere spectrums.

    These spectrums are read from ``.gits.gz`` files in a directory given by
    ``phoenix_path``
    Each file must contain the spectrum for one temperature

    Parameters
    ----------

    phoenix_path: str, **required**
        Path to folder containing phoenix ``fits.gz`` files

    temperature: float, optional
        Stellar temperature in Kelvin

    radius: float, optional
        Stellar radius in Solar radius

    metallicity: float, optional
        Metallicity in solar values

    mass: float, optional
        Stellar mass in solar mass

    distance: float, optional
        Distance from Earth in pc

    magnitudeK: float, optional
        Maginitude in K band


    Raises
    ------
    Exception
        Raised when no phoenix path is defined


    """

    def __init__(
        self,
        temperature: t.Optional[float] = 5000,
        radius: t.Optional[float] = 1.0,
        metallicity: t.Optional[float] = 1.0,
        mass: t.Optional[float] = 1.0,
        distance: t.Optional[float] = 1,
        magnitudeK: t.Optional[float] = 10.0,  # noqa: N803
        phoenix_path: t.Optional[PathLike] = None,
        retro_version_file: t.Optional[PathLike] = None,
    ):
        super().__init__(
            temperature=temperature,
            radius=radius,
            distance=distance,
            magnitudeK=magnitudeK,
            mass=mass,
            metallicity=metallicity,
        )
        self._phoenix_path = phoenix_path

        self.retro_version_file = retro_version_file
        # CAN BE OBTAINED FROM:
        # ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/

        if self._phoenix_path is None or not os.path.isdir(self._phoenix_path):
            self._phoenix_path = GlobalCache()["phoenix_path"]

        if self._phoenix_path is None or not os.path.isdir(self._phoenix_path):
            self.error(
                "No file path or incorrect path to "
                f"phoenix files defined - {self._phoenix_path}"
            )
            raise Exception(
                "No file path or incorrect path "
                f"to phoenix files defined - {self._phoenix_path}"
            )

        self.info("Star is PHOENIX type")

        self.get_avail_phoenix()
        self.use_blackbody = False
        self.recompute_spectra()
        # self.preload_phoenix_spectra()

    def compute_logg(self) -> float:
        """Computes log(surface_G)."""
        import astropy.units as u
        from astropy.constants import G

        mass = self._mass * u.kg
        radius = self._radius * u.m

        small_g = (G * mass) / (radius**2)

        small_g = small_g.to(u.cm / u.s**2)

        return math.log10(small_g.value)

    def recompute_spectra(self) -> None:
        """Recompute spectra as needed."""
        if (
            self.temperature > self._T_list.max()
            or self.temperature < self._T_list.min()
        ):
            self._logg = self.compute_logg()
            self.use_blackbody = True
        else:
            self.use_blackbody = False
            self._logg = self.compute_logg()
            f = self.find_nearest_file()
            self.read_spectra(f)

    def read_spectra(self, p_file: PathLike) -> None:
        """Reads in the spectra from a given file."""
        import astropy.units as u
        from astropy.io import fits

        if self.retro_version_file is not None:
            with fits.open(p_file) as hdu:
                with fits.open(self.retro_version_file) as hdu2:
                    str_unit = hdu2[0].header["UNIT"]
                    wl = hdu2[0].data * u.Unit(str_unit)

                    str_unit = hdu[0].header["BUNIT"]
                    sed = hdu[0].data * u.Unit(str_unit)

                    self.wngrid = 10000 / (wl.to(u.micron).value)
                    argidx = np.argsort(self.wngrid)
                    self._base_sed = sed.to(u.W / u.m**2 / u.micron).value
                    self.wngrid = self.wngrid[argidx]
                    self._base_sed = self._base_sed[argidx]
        else:
            with fits.open(p_file) as hdu:
                str_unit = hdu[1].header["TUNIT1"]
                wl = hdu[1].data.field("Wavelength") * u.Unit(str_unit)

                str_unit = hdu[1].header["TUNIT2"]
                sed = hdu[1].data.field("Flux") * u.Unit(str_unit)

                self.wngrid = 10000 / (wl.value)
                argidx = np.argsort(self.wngrid)
                self._base_sed = sed.to(u.W / u.m**2 / u.micron).value
                self.wngrid = self.wngrid[argidx]
                self._base_sed = self._base_sed[argidx]

    @property
    def temperature(self) -> float:
        """Effective Temperature in Kelvin.

        Returns
        -------
        T: float

        """
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        self._temperature = value
        self.recompute_spectra()

    @property
    def mass(self) -> float:
        """Mass of star in solar mass.

        Returns
        -------
        M: float

        """
        return self._mass

    @mass.setter
    def mass(self, value: float) -> None:
        self._mass = value * MSOL
        self.recompute_spectra()

    def find_nearest_file(self) -> str:
        """Finds the nearest file to the current stellar parameters."""
        idx = self._index_finder([self._temperature, self._logg, self._metallicity])
        return self._files[int(idx)]

    def get_avail_phoenix(self) -> None:
        import glob

        from scipy.interpolate import NearestNDInterpolator

        if self.retro_version_file is not None:
            files = glob.glob(os.path.join(self._phoenix_path, "*.fits"))
        else:
            files = glob.glob(os.path.join(self._phoenix_path, "*.spec.fits.gz"))
        # files = glob.glob(os.path.join(self._phoenix_path, '*.fits'))

        self._files = files
        if self.retro_version_file is not None:
            self._T_list = np.array(
                [np.float64(os.path.basename(k)[3:8]) for k in files]
            )
            self._Logg_list = np.array(
                [np.float64(os.path.basename(k)[9:13]) for k in files]
            )
            self._Z_list = np.array(
                [np.float64(os.path.basename(k)[14:17]) for k in files]
            )
        else:
            self._T_list = (
                np.array([np.float64(os.path.basename(k)[3:8]) for k in files]) * 100
            )
            self._Logg_list = np.array(
                [np.float64(os.path.basename(k)[8:12]) for k in files]
            )
            self._Z_list = np.array(
                [np.float64(os.path.basename(k)[13:16]) for k in files]
            )
        self._index_finder = NearestNDInterpolator(
            (self._T_list, self._Logg_list, self._Z_list),
            np.arange(0, self._T_list.shape[0]),
            rescale=True,
        )

    def initialize(self, wngrid: npt.NDArray[np.float64]):
        """Initializes and interpolates the spectral emission density

        SED is interpolated to the current stellar temperature and
        given wavenumber grid

        Parameters
        ----------
        wngrid:
            Wavenumber grid to interpolate the SED to

        """
        # If temperature outside of range, use blavkbody
        if self.use_blackbody:
            self.warning(
                "Using black body as temperature is "
                "outside of Star temeprature range %s",
                self.temperature,
            )
            super().initialize(wngrid)
        else:
            sed = self._base_sed
            self.sed = np.interp(wngrid, self.wngrid, sed)

    @property
    def spectralEmissionDensity(self) -> float:  # noqa: N802
        """Spectral emmision density in W/m^2/um.

        Returns
        -------
        sed: :obj:`array`
        """
        return self.sed

    def write(self, output: OutputGroup) -> OutputGroup:
        """Write to output group."""
        star = super().write(output)
        star.write_string("phoenix_path", self._phoenix_path)
        return star

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        return ("phoenix",)

    BIBTEX_ENTRIES = [
        r"""
        @article{ refId0,
            author = {{Husser, T.-O.} and {Wende-von Berg, S.} and
            {Dreizler, S.} and {Homeier, D.} and {Reiners, A.} and
            {Barman, T.} and {Hauschildt, P. H.}},
            title = {A new extensive library of PHOENIX stellar atmospheres and
                synthetic spectra},
            DOI= "10.1051/0004-6361/201219058",
            url= "https://doi.org/10.1051/0004-6361/201219058",
            journal = {A\&A},
            year = 2013,
            volume = 553,
            pages = "A6",
            month = "",
        }
        """
    ]

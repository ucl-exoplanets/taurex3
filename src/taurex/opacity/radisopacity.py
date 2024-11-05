"""Opacity module using RADIS code"""
import typing as t
from functools import lru_cache

import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.constants import k_B

from .opacity import Opacity


class RadisHITRANOpacity(Opacity):
    """Radis opacity using HITRAN database.

    This module uses the RADIS code to compute opacity
    using the HITRAN database. This module is not recommended
    for general use as it is very slow and is deprioritized
    in the opacity module list.

    """

    @classmethod
    def priority(cls) -> int:
        """Priority of opacity module."""
        return 1000000

    @classmethod
    def discover(cls) -> t.List[t.Tuple[str, t.Tuple[str, float, float, int]]]:
        from taurex.cache import GlobalCache

        if GlobalCache()["enable_radis"] is not True:
            return []

        trans = {
            "1": "H2O",
            "2": "CO2",
            "3": "O3",
            "4": "N2O",
            "5": "CO",
            "6": "CH4",
            "7": "O2",
            "9": "SO2",
            "10": "NO2",
            "11": "NH3",
            "12": "HNO3",
            "13": "OH",
            "14": "HF",
            "15": "HCl",
            "16": "HBr",
            "17": "HI",
            "18": "ClO",
            "19": "OCS",
            "20": "H2CO",
            "21": "HOCl",
            "23": "HCN",
            "24": "CH3Cl",
            "25": "H2O2",
            "26": "C2H2",
            "27": "C2H6",
            "28": "PH3",
            "29": "COF2",
            "30": "SF6",
            "31": "H2S",
            "32": "HCOOH",
            "33": "HO2",
            "34": "O",
            "35": "ClONO2",
            "36": "NO+",
            "37": "HOBr",
            "38": "C2H4",
            "40": "CH3Br",
            "41": "CH3CN",
            "42": "CF4",
            "43": "C4H2",
            "44": "HC3N",
            "46": "CS",
            "47": "SO3",
        }

        mol_list = trans.values()
        wn_start, wn_end, wn_points = 600, 30000, 10000
        grid = GlobalCache()["radis_grid"]
        if grid is not None:
            wn_start, wn_end, wn_points = grid

        return [(m, [m, wn_start, wn_end, wn_points]) for m in mol_list]

    def __init__(
        self,
        molecule_name: str,
        wn_start: t.Optional[float] = 600,
        wn_end: t.Optional[float] = 30000,
        wn_points: t.Optional[int] = 10000,
    ):
        """Initialize RadisHITRANOpacity.

        Parameters
        ----------
        molecule_name:
            Molecule name.

        wn_start:
            Wavenumber start, by default 600 cm-1.

        wn_end:
            Wavenumber end, by default 30000 cm-1.

        wn_points:
            Wavenumber points, by default 10000.

        """
        super().__init__(self.__class__.__name__)
        import radis

        step = (wn_end - wn_start) / wn_points

        self.info("RADIS Grid set to %s %s %s", wn_start, wn_end, step)

        self.rad_xsec = radis.SpectrumFactory(
            wavenum_min=wn_start,
            wavenum_max=wn_end,
            isotope="1",  # 'all',
            # depends on HAPI benchmark.
            wstep=step,
            verbose=0,
            cutoff=1e-27,
            mole_fraction=1.0,
            # Corresponds to WavenumberWingHW/HWHM=50 in HAPI
            broadening_max_width=10.0,
            molecule=molecule_name,
        )

        self.rad_xsec.fetch_databank("astroquery", load_energies=False)

        self._molecule_name = molecule_name

        s = self.rad_xsec.eq_spectrum(Tgas=296, pressure=1)
        wn, absnce = s.get("abscoeff")

        self.wn = 1e7 / wn

    @property
    def moleculeName(self) -> str:  # noqa: N802
        """Molecule name."""
        return self._molecule_name

    @property
    def wavenumberGrid(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Wavenumber grid."""
        return self.wn

    @property
    def temperatureGrid(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Temperature grid."""
        raise NotImplementedError

    @property
    def pressureGrid(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Pressure grid."""
        raise NotImplementedError

    @lru_cache(maxsize=500)  # noqa: B019
    def compute_opacity(
        self, temperature: float, pressure: float
    ) -> npt.NDArray[np.float64]:
        """Compute opacity.

        Parameters
        ----------
        temperature:
            Temperature in K.

        pressure:
            Pressure in bar.

        Returns
        -------
        opacity:
            Opacity array at K, Pa.

        """
        pressure_pascal = pressure * u.Pascal
        pressure_bar = pressure_pascal.to(u.bar)
        temperature_k = temperature * u.K

        density = ((pressure_bar) / (k_B * temperature_k)).value * 1000

        s = self.rad_xsec.eq_spectrum(Tgas=temperature, pressure=pressure_bar.value)
        wn, absnce = s.get("abscoeff")

        return absnce / density

    BIBTEX_ENTRIES = [
        r"""
        @ARTICLE{2019JQSRT.222...12P,
            author = {{Pannier}, Erwan and {Laux}, Christophe O.},
                title = "{RADIS: A nonequilibrium
                line-by-line radiative code for CO$_{2}$ and
                HITRAN-like database species}",
            journal = {Journal of Quantitative Spectroscopy & Radiative Transfer},
            keywords = {Line-by-line code, Nonequilibrium,
            Optical emission spectroscopy, Absorption spectroscopy},
                year = 2019,
                month = jan,
            volume = {222},
                pages = {12-25},
                doi = {10.1016/j.jqsrt.2018.09.027},
            adsurl = {https://ui.adsabs.harvard.edu/abs/2019JQSRT.222...12P},
            adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }


        """,
    ]

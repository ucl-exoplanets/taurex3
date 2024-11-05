"""Opacity loaded from HDF5 file."""

import pathlib
import typing as t

import numpy as np
import numpy.typing as npt
from astropy.units import UnitConversionError

from taurex.cache import GlobalCache
from taurex.mpi import allocate_as_shared
from taurex.types import PathLike

from .interpolateopacity import InterpModeType, InterpolatingOpacity


class HDF5Opacity(InterpolatingOpacity):
    """Opacities from HDF5 files.

    These usually come from ExoMol. Specifically the ExoMolOP database.

    """

    @classmethod
    def priority(cls) -> int:
        return 5

    @classmethod
    def discover(
        cls,
    ) -> t.List[t.Tuple[str, t.Tuple[pathlib.Path, InterpModeType, bool]]]:
        """Discover opacities from HDF5 files in path."""
        from taurex.cache import GlobalCache

        path = GlobalCache()["xsec_path"]
        if path is None:
            return []

        path = pathlib.Path(path)

        file_list = list(path.glob("*.h5")) + list(path.glob("*.hdf5"))

        discovery = []

        interp = GlobalCache()["xsec_interpolation"] or "linear"
        mem = GlobalCache()["xsec_in_memory"] or True

        for f in file_list:
            op = HDF5Opacity(f, interpolation_mode="linear", in_memory=False)
            mol_name = op.moleculeName
            discovery.append((mol_name, [f, interp, mem]))
            # op._spec_dict.close()
            del op

        return discovery

    def __init__(
        self,
        filename: PathLike,
        interpolation_mode: t.Optional[InterpModeType] = "exp",
        in_memory: t.Optional[bool] = False,
    ) -> None:
        """Initialize opacity from HDF5 file.

        Parameters
        ----------
        filename: PathLike
            Path to opacity file.

        interpolation_mode: str
            Interpolation mode. Either 'linear' or 'exp'.

        in_memory: bool
            If True, the opacity will be loaded in shared memory

        Raises
        ------
        FileNotFoundError
            If file does not exist.

        """
        filename = pathlib.Path(filename)
        super().__init__(
            f"HDF5Opacity:{filename.stem[:10]}",
            interpolation_mode=interpolation_mode,
        )

        if not filename.exists():
            raise FileNotFoundError(f"Could not find {filename}")

        self._filename = filename
        self._molecule_name = None
        self._spec_dict = None
        self.in_memory = in_memory
        self._molecular_citation = []
        self._load_hdf_file(filename)

    @property
    def moleculeName(self) -> str:  # noqa: N802
        """Name of molecule."""
        return self._molecule_name

    @property
    def xsecGrid(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Opacity grid."""
        return self._xsec_grid

    def _load_hdf_file(self, filename: pathlib.Path) -> None:  # noqa: C901
        """Load opacity from HDF5 file.

        Parameters
        ----------
        filename: PathLike
            Path to opacity file.


        """
        import astropy.units as u
        import h5py

        from taurex.data.citation import has_pybtex
        from taurex.util import ensure_string_utf8

        # Load the hdf5
        self.debug(f"Loading opacity from {filename}")

        self._spec_dict = h5py.File(filename, "r")

        self._wavenumber_grid: npt.NDArray[np.float64] = self._spec_dict["bin_edges"][
            ()
        ]

        self._temperature_grid: npt.NDArray[np.float64] = self._spec_dict["t"][()]

        pressure_units = self._spec_dict["p"].attrs["units"]
        try:
            p_conversion = u.Unit(pressure_units).to(u.Pa)
        except UnitConversionError:
            p_conversion = u.Unit(pressure_units, format="cds").to(u.Pa)

        self._pressure_grid: npt.NDArray[np.float64] = (
            self._spec_dict["p"][()] * p_conversion
        )

        if self.in_memory and GlobalCache()["mpi_use_shared"]:
            self._xsec_grid = allocate_as_shared(
                # Dont copy the array until shared memory
                # is allocated. Then copy it to shared memory
                # directly
                self._spec_dict["xsecarr"][()],
                logger=self,
            )
        elif self.in_memory:
            self._xsec_grid = self._spec_dict["xsecarr"][()]
        else:
            self._xsec_grid = self._spec_dict["xsecarr"]

        self._resolution = np.average(np.diff(self._wavenumber_grid))
        self._molecule_name = self._spec_dict["mol_name"][()][()]

        if isinstance(self._molecule_name, np.ndarray):
            self._molecule_name = self._molecule_name[0]

        try:
            self._molecule_name = self._molecule_name.decode()
        except (
            UnicodeDecodeError,
            AttributeError,
        ):
            pass

        self._molecule_name = ensure_string_utf8(self._molecule_name)

        self._min_pressure = self._pressure_grid.min()
        self._max_pressure = self._pressure_grid.max()
        self._min_temperature = self._temperature_grid.min()
        self._max_temperature = self._temperature_grid.max()
        self._molecular_citation = []

        if "DOI" in self._spec_dict:
            doi = self._spec_dict["DOI"][()]
            if isinstance(doi, np.ndarray):
                doi = doi[0]

            molecular_citation = ensure_string_utf8(self._spec_dict["DOI"][()][0])
            self._molecular_citation = [molecular_citation]
        if has_pybtex:
            self.handle_pybtex()

        if self.in_memory:
            self._spec_dict.close()

    def handle_pybtex(self) -> None:
        """Handle citations"""
        from taurex.cache import GlobalCache
        from taurex.data.citation import doi_to_bibtex
        from taurex.util import ensure_string_utf8

        try:
            doi = self._spec_dict["DOI"][()]
            if isinstance(doi, np.ndarray):
                doi = doi[0]

            molecular_citation = ensure_string_utf8(self._spec_dict["DOI"][()][0])
            new_bib = None

            check_xsec = GlobalCache()["xsec_disable_doi"]

            if check_xsec is None:
                check_xsec = False
            else:
                check_xsec = not check_xsec

            if check_xsec:
                new_bib = doi_to_bibtex(molecular_citation)

            self._molecular_citation = [new_bib or molecular_citation]
        except Exception:
            self.warning("DOI could not be determined from hdf5 file.")
            self._molecular_citation = []

    @property
    def wavenumberGrid(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Wavenumber grid."""
        return self._wavenumber_grid

    @property
    def temperatureGrid(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Temperature grid."""
        return self._temperature_grid

    @property
    def pressureGrid(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Pressure grid."""
        return self._pressure_grid

    @property
    def resolution(self) -> float:  # noqa: N802
        """Resolution of opacity."""
        return self._resolution

    def citations(self) -> t.List[str]:
        """Return citations for opacity + ExomolOP."""
        from taurex.data.citation import unique_citations_only

        citations = super().citations()
        opacities = []

        for o in self.opacityCitation():
            try:
                from pybtex.database import Entry

                e = Entry.from_string(o, "bibtex")
            except (IndexError, ImportError):
                e = o
            opacities.append(e)

        citations = citations + opacities
        return unique_citations_only(citations)

    def opacityCitation(self) -> t.List[str]:  # noqa: N802
        """Get citations to opacity."""
        return self._molecular_citation

    BIBTEX_ENTRIES = [
        r"""
    @ARTICLE{2021A&A...646A..21C,
        author = {{Chubb}, Katy L. and {Rocchetto}, Marco and {Yurchenko},
        Sergei N. and {Min}, Michiel and {Waldmann}, Ingo and {Barstow}, Joanna K. and
        {Molli{\`e}re}, Paul and {Al-Refaie}, Ahmed F.
        and {Phillips}, Mark W. and {Tennyson},
        Jonathan},
            title = "{The ExoMolOP database: Cross sections and k-tables for
            molecules of interest in high-temperature exoplanet atmospheres}",
        journal = {Astronomy and Astrophysics},
        keywords = {molecular data, opacity, radiative
        transfer, planets and satellites: atmospheres,
        planets and satellites: gaseous planets,
        infrared: planetary systems, Astrophysics - Earth and Planetary Astrophysics,
        Astrophysics - Instrumentation and Methods for
        Astrophysics, Astrophysics - Solar
        and Stellar Astrophysics},
            year = 2021,
            month = feb,
        volume = {646},
            eid = {A21},
            pages = {A21},
            doi = {10.1051/0004-6361/202038350},
    archivePrefix = {arXiv},
        eprint = {2009.00687},
    primaryClass = {astro-ph.EP},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2021A&A...646A..21C},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

        """
    ]

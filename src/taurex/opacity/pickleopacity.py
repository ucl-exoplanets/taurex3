"""Opacities loaded from pickle."""

import pathlib
import pickle  # noqa
import typing as t

import numpy as np
import numpy.typing as npt

from taurex.cache import GlobalCache
from taurex.mpi import allocate_as_shared
from taurex.mpi import has_mpi
from taurex.mpi import shared_rank
from taurex.types import PathLike
from taurex.types import get_float_dtype
from taurex.util import sanitize_molecule_string

from .interpolateopacity import InterpModeType
from .interpolateopacity import InterpolatingOpacity


class PickleOpacity(InterpolatingOpacity):
    """Class for loading opacities from pickle files.

    This was the original format for TauREx 1/2 opacities. It is now
    deprecated in favour of HDF5 files.

    """

    @classmethod
    def discover(cls) -> t.List[t.Tuple[str, t.Tuple[pathlib.Path, str]]]:
        """Discover opacities from pickle files in path."""
        from taurex.util import sanitize_molecule_string

        path = GlobalCache()["xsec_path"]
        if path is None:
            return []
        path = pathlib.Path(path)

        files = path.glob("*.pickle")

        discovery = []

        interp = GlobalCache()["xsec_interpolation"] or "linear"

        for f in files:
            splits = f.stem.split(".")
            mol_name = sanitize_molecule_string(splits[0])

            discovery.append((mol_name, [f, interp]))

        return discovery

    def __init__(
        self,
        filename: PathLike,
        interpolation_mode: t.Optional[InterpModeType] = "linear",
    ) -> None:
        """Initialize and load pickle file.

        Parameters
        ----------
        filename: PathLike
            Path to pickle file

        interpolation_mode:
            Interpolation mode, by default "linear"

        Raises
        ------
        FileNotFoundError
            If file is not found

        """
        filename = pathlib.Path(filename)
        super().__init__(
            f"PickleOpacity:{filename.stem[0:10]}",
            interpolation_mode=interpolation_mode,
        )
        import warnings

        warnings.warn(
            "PickleOpacity is deprecated and will be removed"
            " in a future major version. "
            "Please use HDF5Opacity instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not filename.exists():
            raise FileNotFoundError(f"File {filename} does not exist")
        self._filename = filename
        self._molecule_name = None
        self._spec_dict = None
        self._resolution = None
        self._load_pickle_file(filename)

    @property
    def moleculeName(self) -> str:  # noqa: N802
        """Molecule name."""
        return self._molecule_name

    @property
    def xsecGrid(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Opacity grid."""
        return self._xsec_grid

    def _load_pickle_file(self, filename: pathlib.Path) -> None:
        """Load pickle file into memory."""
        use_shared = bool(GlobalCache()["mpi_use_shared"])
        sh_root = (not has_mpi()) or shared_rank() == 0

        if use_shared and not sh_root:
            # Non-root ranks in a shared-memory group avoid loading
            # the large pickle payload and attach directly to the
            # shared allocations created by rank 0.
            xsec_arr = None
            wn_arr = None
            temp_arr = None
            press_arr = None
        else:
            # Load the pickle file
            self.info(f"Loading opacity from {filename}")
            try:
                with open(filename, "rb") as f:
                    self._spec_dict = pickle.load(f)  # noqa
            except UnicodeDecodeError:
                with open(filename, "rb") as f:
                    self._spec_dict = pickle.load(f, encoding="latin1")  # noqa

            self._wavenumber_grid = self._spec_dict["wno"]
            self._temperature_grid = self._spec_dict["t"]
            self._pressure_grid = self._spec_dict["p"] * 1e5

            # Extract arrays and free the raw pickle dict immediately.
            # This avoids holding two copies during the shared-memory copy.
            xsec_arr = self._spec_dict["xsecarr"]
            wn_arr = self._wavenumber_grid
            temp_arr = self._temperature_grid
            press_arr = self._pressure_grid
            self._spec_dict = None

        self._xsec_grid = allocate_as_shared(xsec_arr, logger=self)

        # Also share the auxiliary grid arrays when using MPI shared memory.
        if use_shared:
            self._wavenumber_grid = allocate_as_shared(wn_arr, logger=self)
            self._temperature_grid = allocate_as_shared(temp_arr, logger=self)
            self._pressure_grid = allocate_as_shared(press_arr, logger=self)
        else:
            self._wavenumber_grid = wn_arr
            self._temperature_grid = temp_arr
            self._pressure_grid = press_arr

        # ---- float32_mode: cast all loaded arrays to float32 if enabled ----
        if GlobalCache()["float32_mode"]:
            _dt = get_float_dtype()
            self._wavenumber_grid = self._wavenumber_grid.astype(_dt)
            self._temperature_grid = self._temperature_grid.astype(_dt)
            self._pressure_grid = self._pressure_grid.astype(_dt)
            self._xsec_grid = self._xsec_grid.astype(_dt)
        # --------------------------------------------------------------------

        self._resolution = np.average(np.diff(self._wavenumber_grid))

        splits = filename.stem.split(".")
        mol_name = sanitize_molecule_string(splits[0])
        self._molecule_name = mol_name

        self._min_pressure = self._pressure_grid.min()
        self._max_pressure = self._pressure_grid.max()
        self._min_temperature = self._temperature_grid.min()
        self._max_temperature = self._temperature_grid.max()
        self.clean_molecule_name()

    def clean_molecule_name(self) -> None:
        """Clean molecule name from underscores used."""
        splits = self.moleculeName.split("_")
        self._molecule_name = splits[0]

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

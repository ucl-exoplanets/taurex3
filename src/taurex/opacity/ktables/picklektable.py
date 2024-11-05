"""Ktables from pickle files."""
import pathlib
import pickle  # noqa: S403
import typing as t

import numpy as np
import numpy.typing as npt

from taurex.types import PathLike

from ..interpolateopacity import InterpModeType, InterpolatingOpacity
from .ktable import KTable


class PickleKTable(KTable, InterpolatingOpacity):
    """Ktables loaded from pickle files.

    This is the old way of loading ktables from pickle files
    from TauREx 1/2. It is not recommended to use this format
    of loading ktables as it is not as flexible as the HDF5
    format. It is recommended to use the HDF5 format instead.

    """

    @classmethod
    def discover(cls) -> t.List[t.Tuple[str, t.Tuple[pathlib.Path, InterpModeType]]]:
        import pathlib

        from taurex.cache import GlobalCache
        from taurex.util import sanitize_molecule_string

        path = GlobalCache()["ktable_path"]
        if path is None:
            return []
        path = pathlib.Path(path)

        files = path.glob("*.pickle")

        discovery = []

        interp = GlobalCache()["xsec_interpolation"] or "linear"

        for f in files:
            splits = pathlib.Path(f).stem.split(".")
            mol_name = sanitize_molecule_string(splits[0])

            discovery.append((mol_name, [f, interp]))

        return discovery

    def __init__(
        self, filename: PathLike, interpolation_mode: t.Optional[PathLike] = "linear"
    ) -> None:
        """Initialize PickleKTable.

        Parameters
        ----------
        filename : PathLike
            Path to file.
        interpolation_mode : t.Optional[PathLike], optional
            Interpolation mode, by default "linear".


        """
        filename = pathlib.Path(filename)
        super().__init__(
            f"PickleKtable:{filename.stem[0:10]}",
            interpolation_mode=interpolation_mode,
        )
        if not filename.exists():
            raise FileNotFoundError(f"File {filename} not found")

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
        """Cross section grid."""
        return self._xsec_grid

    def _load_pickle_file(self, filename: pathlib.Path) -> None:
        """Load pickle file."""
        self.info(f"Loading opacity from {filename}")
        try:
            with open(filename, "rb") as f:
                self._spec_dict = pickle.load(f)  # noqa: S301
        except UnicodeDecodeError:
            with open(filename, "rb") as f:
                self._spec_dict = pickle.load(f, encoding="latin1")  # noqa: S301

        self._wavenumber_grid = self._spec_dict["bin_centers"]
        self._ngauss = self._spec_dict["ngauss"]
        self._temperature_grid = self._spec_dict["t"]
        self._pressure_grid = self._spec_dict["p"] * 1e5
        self._xsec_grid = self._spec_dict["kcoeff"]
        self._weights = self._spec_dict["weights"]
        self._molecule_name = self._spec_dict["name"]

        self._min_pressure = self._pressure_grid.min()
        self._max_pressure = self._pressure_grid.max()
        self._min_temperature = self._temperature_grid.min()
        self._max_temperature = self._temperature_grid.max()
        self.clean_molecule_name()

    def clean_molecule_name(self) -> None:
        """Clean molecule name."""
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
    def resolution(self) -> float:
        """Resolution."""
        return self._resolution

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Quadrature Weights."""
        return self._weights

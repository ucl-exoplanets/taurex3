"""Class for handling ktables from NEMESIS."""
import pathlib
import typing as t

import numpy as np
import numpy.typing as npt

from taurex.types import PathLike
from taurex.util import sanitize_molecule_string

from ..interpolateopacity import InterpModeType, InterpolatingOpacity
from .ktable import KTable


class NemesisKTables(KTable, InterpolatingOpacity):
    """KTables from NEMESIS."""

    @classmethod
    def discover(
        cls,
    ) -> t.Optional[t.List[t.Tuple[str, t.Tuple[pathlib.Path, InterpModeType]]]]:
        """Discover NEMESIS ktables in path."""
        import pathlib

        from taurex.cache import GlobalCache

        path = GlobalCache()["ktable_path"]
        if path is None:
            return []
        path = pathlib.Path(path)

        files = path.glob("*.kta")

        discovery = []

        interp = GlobalCache()["xsec_interpolation"] or "linear"

        for f in files:
            splits = pathlib.Path(f).stem.split("_")
            mol_name = sanitize_molecule_string(splits[0])

            discovery.append((mol_name, [f, interp]))

        return discovery

    def __init__(
        self,
        filename: PathLike,
        interpolation_mode: t.Optional[InterpModeType] = "linear",
    ):
        """Initialize and read NEMESIS ktable.

        Parameters
        ----------
        filename : PathLike
            Path to file.
        interpolation_mode : t.Optional[InterpModeType], optional
            Interpolation mode, by default "linear".


        """
        filename = pathlib.Path(filename)
        super().__init__(
            f"NemesisKtable:{filename.stem[0:10]}",
            interpolation_mode=interpolation_mode,
        )
        if not filename.exists():
            raise FileNotFoundError(f"Could not find file {filename}")
        self._filename = filename
        splits = filename.stem.split("_")
        mol_name = sanitize_molecule_string(splits[0])
        self._molecule_name = mol_name
        self._spec_dict = None
        self._resolution = None
        self._decode_ktables(filename)

    @property
    def moleculeName(self) -> str:  # noqa: N802
        """Molecule name."""
        return self._molecule_name

    @property
    def xsecGrid(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Cross section grid."""
        return self._xsec_grid

    def _decode_ktables(self, filename: pathlib.Path):
        """Decode NEMESIS ktable format."""
        self.debug("Reading NEMESIS FORMAT")
        nem_file_float = np.fromfile(filename, dtype=np.float32)
        nem_file_int = nem_file_float.view(np.int32)
        array_counter = 0
        self.debug("MAGIC NUMBER: %s", nem_file_int[0])
        wncount = nem_file_int[1]
        self.debug("WNCOUNT = %s", wncount)
        wnstart = nem_file_float[2]
        self.debug("WNSTART = %s um", wnstart)
        float_num = nem_file_float[3]
        self.debug("FLOAT: %s INT: %s", float_num, nem_file_int[4])

        num_pressure = nem_file_int[5]
        num_temperature = nem_file_int[6]
        num_quads = nem_file_int[7]

        self.debug("NP: %s NT: %s NQ: %s", num_pressure, num_temperature, num_quads)
        self.debug("UNKNOWN VALUES: %s %s", nem_file_int[8], nem_file_int[9])

        array_counter += 10 + num_quads * 2
        self._samples, self._weights = (
            nem_file_float[10:array_counter].reshape(2, -1).astype(np.float64)
        )
        self.debug("Samples: %s, Weights: %s", self._samples, self._weights)
        self.debug("%s", nem_file_int[array_counter])
        array_counter += 1
        self.debug("%s", nem_file_int[array_counter])
        array_counter += 1
        self._pressure_grid = (
            nem_file_float[array_counter : array_counter + num_pressure].astype(
                np.float64
            )
            * 1e5
        )
        self.debug("Pgrid: %s", self._pressure_grid)
        array_counter += num_pressure
        self._temperature_grid = nem_file_float[
            array_counter : array_counter + num_temperature
        ].astype(np.float64)
        array_counter += num_temperature
        self.debug("Tgrid: %s", self._temperature_grid)
        self._wavenumber_grid = 10000 / nem_file_float[
            array_counter : array_counter + wncount
        ].astype(np.float64)
        self._wavenumber_grid = self._wavenumber_grid[::-1]
        array_counter += wncount
        self.debug("Wngrid: %s", self._wavenumber_grid)

        self._xsec_grid = (
            nem_file_float[array_counter:].reshape(
                wncount, num_pressure, num_temperature, num_quads
            )
            * 1e-20
        ).astype(np.float64)
        self._xsec_grid = self._xsec_grid.transpose((1, 2, 0, 3))
        self._xsec_grid = self._xsec_grid[::, ::, ::-1, :]
        self._min_pressure = self._pressure_grid.min()
        self._max_pressure = self._pressure_grid.max()
        self._min_temperature = self._temperature_grid.min()
        self._max_temperature = self._temperature_grid.max()

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
        """Resolution of grid."""
        return self._resolution

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Quandrature Weights."""
        return self._weights

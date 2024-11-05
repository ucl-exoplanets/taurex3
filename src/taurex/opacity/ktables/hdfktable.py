"""Ktables loaded from HDF5."""
import pathlib
import typing as t

import h5py
import numpy as np
import numpy.typing as npt

from taurex.types import PathLike
from taurex.util import sanitize_molecule_string

from ..interpolateopacity import InterpModeType, InterpolatingOpacity
from .ktable import KTable


class HDF5KTable(KTable, InterpolatingOpacity):
    """
    This is the base class for computing opactities using correlated k tables
    """

    @classmethod
    def discover(cls) -> t.List[t.Tuple[str, t.Tuple[pathlib.Path, str]]]:
        """Discover opacities from hdf5 files in path."""
        from taurex.cache import GlobalCache

        path = GlobalCache()["ktable_path"]
        if path is None:
            return []
        path = pathlib.Path(path)

        files = list(path.glob("*.h5")) + list(path.glob("*.hdf5"))

        discovery = []

        interp = GlobalCache()["xsec_interpolation"] or "linear"

        for f in files:
            splits = f.stem.split("_")
            mol_name = sanitize_molecule_string(splits[0])

            discovery.append((mol_name, [f, interp]))

        return discovery

    def __init__(
        self,
        filename: PathLike,
        interpolation_mode: t.Optional[InterpModeType] = "linear",
        in_memory: t.Optional[bool] = True,
    ) -> None:
        """Initialize HDF5KTable.

        Parameters
        ----------
        filename : PathLike
            Path to file.
        interpolation_mode : t.Optional[InterpModeType], optional
            Interpolation mode, by default "linear".
        in_memory : t.Optional[bool], optional
            Load into memory, by default True.

        Raises
        ------
        FileNotFoundError
            If file not found.


        """
        filename = pathlib.Path(filename)
        self._molecule_name = sanitize_molecule_string(filename.stem.split("_")[0])
        super().__init__(
            f"HDF5Ktable:{self._molecule_name}",
            interpolation_mode=interpolation_mode,
        )

        if not filename.exists():
            raise FileNotFoundError(f"Could not find {filename}")

        self._filename = filename
        self._spec_dict = None

        self._resolution = None
        self.in_memory = in_memory
        self._load_hdf_file(filename)

    @property
    def moleculeName(self) -> str:  # noqa: N802
        """Molecule name."""
        return self._molecule_name

    @property
    def xsecGrid(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Cross section grid."""
        return self._xsec_grid

    def _load_hdf_file(self, filename: pathlib.Path):
        import astropy.units as u

        # Load the hdf file
        self.info(f"Loading opacity from {filename}")

        self._spec_dict = h5py.File(filename, "r")

        self._wavenumber_grid: npt.NDArray[np.float64] = self._spec_dict[
            "bin_centers"
        ].copy()
        self._ngauss: int = self._spec_dict["ngauss"][()]
        self._temperature_grid: npt.NDArray[np.float64] = self._spec_dict["t"].copy()

        pressure_units: npt.NDArray[np.float64] = self._spec_dict["p"].attrs["units"]
        try:
            p_conversion = u.Unit(pressure_units).to(u.Pa)
        except u.UnitConversionError:
            p_conversion = u.Unit(pressure_units, format="cds").to(u.Pa)

        self._pressure_grid: npt.NDArray[np.float64] = (
            self._spec_dict["p"].copy() * p_conversion
        )

        if self.in_memory:
            self._xsec_grid = self._spec_dict["kcoeff"].copy()
        else:
            self._xsec_grid = self._spec_dict["kcoeff"]
        self._weights = self._spec_dict["weights"].copy()

        self._min_pressure = self._pressure_grid.min()
        self._max_pressure = self._pressure_grid.max()
        self._min_temperature = self._temperature_grid.min()
        self._max_temperature = self._temperature_grid.max()
        self.clean_molecule_name()
        if self.in_memory:
            self._spec_dict.close()

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
        """Weights."""
        return self._weights

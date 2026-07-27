"""Handling of .pickle/.db CIA files."""

import pickle  # noqa: S403
import typing as t
from pathlib import Path

import numpy as np
import numpy.typing as npt

from taurex.util.math import interp_lin_only

from .cia import CIA


class PickleCIA(CIA):
    """Class for using pickled (``.db``) collisionally induced absorptions.

    Very simple since the format is simple


    Parameters
    ----------
    filename : str
        Path to pickle

    pair_name : str , optional
        Whilst the name of the pair is determined by the pickle filename
        since these can be different you can optionally force the name through
        this parameter

    """

    def __init__(
        self,
        filename: str,
        pair_name: t.Optional[t.Union[str, None]] = None,
    ):
        """Initialize PickleCIA.

        Parameters
        ----------
        filename : str
            Path to pickle
        pair_name : str , optional
            Whilst the name of the pair is determined by the pickle filename
            since these can be different you can optionally force the name
            through this parameter
        """
        if pair_name is None:
            pair_name = Path(filename).stem

        super().__init__("PickleCIA", pair_name)

        self._filename = filename
        self._molecule_name = None
        self._spec_dict = None
        self._load_pickle_file(filename)

    def _load_pickle_file(self, filename: str) -> None:
        """Loads pickle file.

        Parameters
        ----------
        filename : str
            Path to pickle cia file

        """
        from taurex.cache import GlobalCache
        from taurex.mpi import allocate_as_shared
        from taurex.mpi import has_mpi
        from taurex.mpi import shared_rank

        use_shared = bool(GlobalCache()["mpi_use_shared"])
        sh_root = (not has_mpi()) or shared_rank() == 0

        if use_shared and not sh_root:
            xsec_arr = None
            wn_arr = None
            temp_arr = None
        else:
            # Load the pickle file
            self.info("Loading cia cross section from %s", filename)
            with open(filename, "rb") as f:
                self._spec_dict = pickle.load(f, encoding="latin1")  # noqa: S301

            # Extract arrays and free the pickle dict immediately.
            # This avoids holding two copies (pickle dict + extracted
            # arrays) simultaneously during the shared-memory copy below.
            xsec_arr = self._spec_dict["xsecarr"]
            wn_arr = self._spec_dict["wno"]
            temp_arr = self._spec_dict["t"]
            self._spec_dict = None

        # Move CIA cross-section array into MPI shared memory so that
        # all ranks on the same node share one copy instead of each
        # loading their own.
        if use_shared:
            self._xsec_grid = allocate_as_shared(xsec_arr, logger=self)
            self._wavenumber_grid = allocate_as_shared(wn_arr, logger=self)
            self._temperature_grid = allocate_as_shared(temp_arr, logger=self)
        else:
            self._xsec_grid = xsec_arr
            self._wavenumber_grid = wn_arr
            self._temperature_grid = temp_arr

    @property
    def wavenumberGrid(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Native wavenumber grid.

        Returns
        -------
        :obj:`array`
            Native wavenumber grid

        """
        return self._wavenumber_grid

    @property
    def temperatureGrid(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Native temperature grid in Kelvin.

        Returns
        -------
        :obj:`array`
            Native temperature grid in Kelvin

        """
        return self._temperature_grid

    def find_closest_temperature_index(self, temperature: float) -> t.Tuple[int, int]:
        """Finds the nearest indices for a particular temperature.

        Parameters
        ----------
        temperature : float
            Temeprature in Kelvin

        Returns
        -------
        t_min : int
            index on temprature grid to the left of ``temperature``

        t_max : int
            index on temprature grid to the right of ``temperature``

        """
        from taurex.util import find_closest_pair

        t_min, t_max = find_closest_pair(self.temperatureGrid, temperature)
        return t_min, t_max

    def interp_linear_grid(
        self, temperature: float, t_idx_min: int, t_idx_max: int
    ) -> float:
        """For a given temperature and indicies.

        Interpolate the cross-sections
        linearly from temperature grid to temperature ``T``

        Parameters
        ----------
        temperature : float
            Temeprature in Kelvin

        t_idx_min : int
            index on temprature grid to the left of ``temperature``

        t_idx_max : int
            index on temprature grid to the right of ``temperature``

        Returns
        -------
        out : :obj:`array`
            Interpolated cross-section

        """
        if temperature > self._temperature_grid.max():
            return self._xsec_grid[-1]
        elif temperature < self._temperature_grid.min():
            return self._xsec_grid[0]

        temp_max = self._temperature_grid[t_idx_max]
        temp_min = self._temperature_grid[t_idx_min]
        fx0 = self._xsec_grid[t_idx_min]
        fx1 = self._xsec_grid[t_idx_max]

        return interp_lin_only(fx0, fx1, temperature, temp_min, temp_max)

    def compute_cia(self, temperature: float) -> npt.NDArray[np.float64]:
        """Computes the collisionally induced absorption cross-section.

        Uses our native temperature and cross-section grids

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin

        Returns
        -------
        out : :obj:`array`
            Temperature interpolated cross-section

        """
        indicies = self.find_closest_temperature_index(temperature)
        return self.interp_linear_grid(temperature, *indicies)

"""Handling of .pickle/.db CIA files."""
import pickle  # noqa: S403
import typing as t
from pathlib import Path

import numpy as np
import numpy.typing as npt

from taurex.util.math import interp_lin_only

from .cia import CIA


class PickleCIA(CIA):
    """
    Class for using pickled (``.db``) collisionally induced absorptions
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

    def __init__(self, filename: str, pair_name: t.Optional[np.float64] = None):
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

        # Load the pickle file
        self.info("Loading cia cross section from %s", filename)
        with open(filename, "rb") as f:
            self._spec_dict = pickle.load(f, encoding="latin1")  # noqa: S301

        self._wavenumber_grid = self._spec_dict["wno"]
        self._temperature_grid = self._spec_dict["t"]
        self._xsec_grid = self._spec_dict["xsecarr"]

    @property
    def wavenumberGrid(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """

        Returns
        -------
        :obj:`array`
            Native wavenumber grid

        """
        return self._wavenumber_grid

    @property
    def temperatureGrid(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """

        Returns
        -------
        :obj:`array`
            Native temperature grid in Kelvin

        """

        return self._temperature_grid

    def find_closest_temperature_index(self, temperature: float) -> t.Tuple[int, int]:
        """
        Finds the nearest indices for a particular temperature

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
        """
        For a given temperature and indicies. Interpolate the cross-sections
        linearly from temperature grid to temperature ``T``

        Parameters
        ----------
        temperature : float
            Temeprature in Kelvin

        t_min : int
            index on temprature grid to the left of ``temperature``

        t_max : int
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
        """
        Computes the collisionally induced absorption cross-section
        using our native temperature and cross-section grids

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

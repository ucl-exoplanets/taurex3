"""Temperature profile loaded from file."""
import typing as t

import numpy as np

from taurex.types import PathLike
from taurex.util import conversion_factor

from .temparray import TemperatureArray


class TemperatureFile(TemperatureArray):
    """A temperature profile read from file.

    If pressure is included in file then the temperature profile
    will be interpolated to the pressure points of the atmosphere.

    """

    def __init__(
        self,
        filename: t.Optional[PathLike] = None,
        skiprows: t.Optional[int] = 0,
        temp_col: t.Optional[int] = 0,
        press_col: t.Optional[int] = None,
        temp_units: t.Optional[str] = "K",
        press_units: t.Optional[str] = "Pa",
        delimiter: t.Optional[str] = None,
        reverse: t.Optional[bool] = False,
    ):
        """Initialize temperature profile from file.

        Parameters
        ----------
        filename : str
            File name for temperature profile
        skiprows : int, optional
            Number of rows to skip
        temp_col : int, optional
            Column number for temperature
        press_col : int, optional
            Column number for pressure
        temp_units : str, optional
            Temperature units
        press_units : str, optional
            Pressure units
        delimiter : str, optional
            Delimiter
        reverse : bool, optional
            Reverse the order of the array

        """
        pressure_arr = None
        temperature_arr = None

        convert_t = conversion_factor(temp_units, "K")
        convert_p = conversion_factor(press_units, "Pa")

        if press_col is not None:
            arr = np.loadtxt(
                filename,
                skiprows=skiprows,
                usecols=(int(press_col), int(temp_col)),
                delimiter=delimiter,
            )
            temperature_arr = arr[:, 1] * convert_t
            pressure_arr = arr[:, 0] * convert_p
        else:
            arr = np.loadtxt(
                filename,
                skiprows=skiprows,
                usecols=int(temp_col),
            )
            temperature_arr = arr[:] * convert_t

        super().__init__(tp_array=temperature_arr, p_points=pressure_arr)

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Return all input keywords."""
        return (
            "file",
            "fromfile",
        )

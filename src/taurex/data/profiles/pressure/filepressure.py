"""Pressure Profile from file"""
import typing as t

import numpy as np

from taurex.types import PathLike
from taurex.util import conversion_factor

from .arraypressure import ArrayPressureProfile


class FilePressureProfile(ArrayPressureProfile):
    """Load pressure profile from file."""

    def __init__(
        self,
        filename: t.Optional[PathLike] = None,
        usecols: t.Optional[int] = 0,
        skiprows: t.Optional[int] = 0,
        units: t.Optional[str] = "Pa",
        delimiter: t.Optional[str] = None,
        reverse: t.Optional[bool] = False,
        is_central: t.Optional[bool] = True,
    ):
        """Initialize the pressure profile.

        Parameters
        ----------
        filename : t.Optional[PathLike], optional
            Filename to load, by default None
        usecols : t.Optional[int], optional
            Which column to use, by default 0
        skiprows : t.Optional[int], optional
            How many rows to skip, by default 0
        units : t.Optional[str], optional
            Units of the pressure, by default "Pa"
        delimiter : t.Optional[str], optional
            Delimiter to use, by default None
        reverse : t.Optional[bool], optional
            Reverse the pressure profile, by default False
        is_central : t.Optional[bool], optional
            If the pressure profile is central or not, by default True
            **New in v3.2.0**


        """
        to_pa = conversion_factor(units, "Pa")

        read_arr = np.loadtxt(
            filename,
            usecols=int(usecols),
            skiprows=int(skiprows),
            delimiter=delimiter,
            dtype=np.float64,
        )
        super().__init__(read_arr * to_pa, reverse=reverse, is_central=is_central)

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        return (
            "file",
            "fromfile",
            "loadfile",
        )

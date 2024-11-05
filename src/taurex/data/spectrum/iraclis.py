"""Spectra from Iraclis pickle data."""
import pickle  # noqa: S403
import typing as t

import numpy as np

from taurex.types import PathLike

from .array import ArraySpectrum


class IraclisSpectrum(ArraySpectrum):
    """Loads an observation from Iraclis pickle data."""

    def __init__(self, filename: t.Optional[PathLike] = None):
        """Initialize.

        Parameters
        ----------
        filename:
            Filename of Iraclis pickle data, by default None


        """
        self._filename = filename
        try:
            with open(filename, "rb") as f:
                database = pickle.load(f)  # noqa: S301
        except UnicodeDecodeError:
            with open(filename, "rb") as f:
                database = pickle.load(f, encoding="latin1")  # noqa: S301

        wl = database["spectrum"]["wavelength"]
        td = database["spectrum"]["depth"]
        err = database["spectrum"]["error"]
        width = database["spectrum"]["width"]

        final_array = np.vstack((wl, td, err, width)).T

        super().__init__(final_array)

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        return ("iraclis",)

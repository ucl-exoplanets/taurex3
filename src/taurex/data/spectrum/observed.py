"""Module for loading an observed spectrum from a text file.

Spectrum must be 3-4 columns with ordering:
    1. wavelength
    2. spectral data
    3. error
    4. (optional) bin width

If no bin width is present then they are computed.
"""

import typing as t

import numpy as np

from .array import ArraySpectrum


class ObservedSpectrum(ArraySpectrum):
    """Loads observed spectrum from a text file.

    Loads an observation and also computes bin edges and bin widths.

    Spectrum must be 3-4 columns with ordering:
        1. wavelength
        2. spectral data
        3. error
        4. (optional) bin width

    If no bin width is present then they are computed.

    Parameters
    ----------
    filename: string
        Path to observed spectrum file.

    """

    def __init__(self, filename=None):
        """Initialize ObservedSpectrum."""
        self._filename = filename

        data = np.loadtxt(self._filename)
        super().__init__(data)

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Input keywords for observed spectrum."""
        return ["dat-file", "observed", "text"]

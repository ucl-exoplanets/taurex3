"""Modules dealing with reading data from observations."""

from .array import ArraySpectrum
from .observed import ObservedSpectrum
from .offsetspectrum import OffsetSpectra
from .offsetspectrum import OffsetSpectraCont
from .spectrum import BaseSpectrum
from .taurex import TaurexSpectrum


__all__ = [
    "BaseSpectrum",
    "ObservedSpectrum",
    "ArraySpectrum",
    "OffsetSpectra",
    "OffsetSpectraCont",
    "TaurexSpectrum",
]

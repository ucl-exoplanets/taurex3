"""Modules dealing with reading data from observations."""

from .array import ArraySpectrum
from .observed import ObservedSpectrum
from .spectrum import BaseSpectrum
from .taurex import TaurexSpectrum

__all__ = ["BaseSpectrum", "ObservedSpectrum", "ArraySpectrum", "TaurexSpectrum"]

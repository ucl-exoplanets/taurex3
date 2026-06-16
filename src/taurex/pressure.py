"""Alias for pressure profiles."""

from taurex.data.profiles.pressure import ArrayPressureProfile
from taurex.data.profiles.pressure import FilePressureProfile
from taurex.data.profiles.pressure import PressureProfile
from taurex.data.profiles.pressure import SimplePressureProfile


__all__ = [
    "PressureProfile",
    "SimplePressureProfile",
    "FilePressureProfile",
    "ArrayPressureProfile",
]

"""Alias for pressure profiles."""

from taurex.data.profiles.pressure import (
    ArrayPressureProfile,
    FilePressureProfile,
    PressureProfile,
    SimplePressureProfile,
)

__all__ = [
    "PressureProfile",
    "SimplePressureProfile",
    "FilePressureProfile",
    "ArrayPressureProfile",
]

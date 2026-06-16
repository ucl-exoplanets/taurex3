"""Alias for temperature profiles."""

from taurex.data.profiles.temperature import Guillot2010
from taurex.data.profiles.temperature import Isothermal
from taurex.data.profiles.temperature import NPoint
from taurex.data.profiles.temperature import Rodgers2000
from taurex.data.profiles.temperature import TemperatureFile
from taurex.data.profiles.temperature import TemperatureProfile


__all__ = [
    "TemperatureProfile",
    "Isothermal",
    "Guillot2010",
    "NPoint",
    "Rodgers2000",
    "TemperatureFile",
]

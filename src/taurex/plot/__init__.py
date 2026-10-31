"""Plotting module for TauREx."""

from .comparison import plot_gas_profile_files
from .comparison import plot_posterior_files
from .comparison import plot_spectrum_files
from .comparison import plot_temperature_pressure_files
from .profile import plot_depth_contrib
from .profile import plot_gas_profile
from .profile import plot_optical_depth
from .profile import plot_temperature_pressure
from .spectrum import plot_spectral_contributions
from .spectrum import plot_spectrum


__all__ = [
    "plot_temperature_pressure_files",
    "plot_gas_profile_files",
    "plot_spectrum_files",
    "plot_posterior_files",
    "plot_optical_depth",
    "plot_depth_contrib",
    "plot_temperature_pressure",
    "plot_gas_profile",
    "plot_spectrum",
    "plot_spectral_contributions",
]

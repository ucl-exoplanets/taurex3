"""Isothermal temperature profile."""
import typing as t

import numpy as np
import numpy.typing as npt

from taurex.data.fittable import fitparam
from taurex.output import OutputGroup

from .tprofile import TemperatureProfile


class Isothermal(TemperatureProfile):
    """An isothermal temperature-pressure profile."""

    def __init__(self, T: t.Optional[float] = 1500) -> None:  # noqa: N803
        """Initialize isothermal class

        Parameters
        ----------

        T : float
            Isothermal Temperature to set in Kelvin
        """
        super().__init__("Isothermal")

        self._iso_temp = T

    @fitparam(
        param_name="T",
        param_latex="$T$",
        default_fit=False,
        default_bounds=[300.0, 2000.0],
    )
    def isoTemperature(self) -> float:  # noqa: N802
        """Isothermal temperature in Kelvin"""
        return self._iso_temp

    @isoTemperature.setter
    def isoTemperature(self, value: float) -> None:  # noqa: N802
        self._iso_temp = value

    @property
    def profile(self) -> npt.NDArray[np.float64]:
        """Returns an isothermal temperature profile."""
        return np.full(self.nlayers, self._iso_temp, dtype=np.float64)

    def write(self, output: OutputGroup) -> OutputGroup:
        """Write isothermal temperature profile to output group."""
        temperature = super().write(output)
        temperature.write_scalar("T", self._iso_temp)
        return temperature

    @classmethod
    def input_keywords(cls) -> t.Tuple[str]:
        """Return all input keywords."""
        return ("isothermal",)

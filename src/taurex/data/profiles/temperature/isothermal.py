"""Isothermal temperature profile."""

import typing as t

import numpy as np
import numpy.typing as npt
from astropy import units as u

from taurex.data.fittable import fitparam
from taurex.output import OutputGroup
from taurex.types import get_float_dtype
from taurex.util import convert_to_unit_value

from .tprofile import TemperatureProfile


class Isothermal(TemperatureProfile):
    """An isothermal temperature-pressure profile."""

    def __init__(
        self, T: t.Optional[t.Union[float, u.Quantity]] = 1500  # noqa: N803
    ) -> None:
        """Initialize isothermal class.

        Parameters
        ----------
        T : float or astropy.units.Quantity
            Isothermal temperature in K when unitless.
        """
        super().__init__("Isothermal")

        self._iso_temp = convert_to_unit_value(
            T, u.K, default_unit=u.K, equivalencies=u.temperature()
        )

    @fitparam(
        param_name="T",
        param_latex="$T$",
        default_fit=False,
        default_bounds=[300.0, 2000.0],
    )
    def isoTemperature(self) -> float:  # noqa: N802
        """Isothermal temperature in Kelvin."""
        return self._iso_temp

    @isoTemperature.setter
    def isoTemperature(  # noqa: N802
        self, value: t.Union[float, u.Quantity]
    ) -> None:
        """Set the isothermal temperature in K when unitless."""
        self._iso_temp = convert_to_unit_value(
            value, u.K, default_unit=u.K, equivalencies=u.temperature()
        )

    @property
    def profile(self) -> npt.NDArray[np.float64]:
        """Returns an isothermal temperature profile."""
        return np.full(self.nlayers, self._iso_temp, dtype=get_float_dtype())

    def write(self, output: OutputGroup) -> OutputGroup:
        """Write isothermal temperature profile to output group."""
        temperature = super().write(output)
        temperature.write_scalar("T", self._iso_temp)
        return temperature

    @classmethod
    def input_keywords(cls) -> t.Tuple[str]:
        """Return all input keywords."""
        return ("isothermal",)

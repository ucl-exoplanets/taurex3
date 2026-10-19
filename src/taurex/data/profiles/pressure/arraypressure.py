"""Pressure profile from array."""

import typing as t

import numpy as np
import numpy.typing as npt
from astropy import units as u

from taurex.output import OutputGroup
from taurex.types import get_float_dtype
from taurex.util import convert_to_unit_value

from .pressureprofile import PressureProfile


class ArrayPressureProfile(PressureProfile):
    """Pressure profile from an array for each layer."""

    def __init__(
        self,
        array: t.Union[npt.ArrayLike, u.Quantity],
        reverse: t.Optional[bool] = False,
        is_central: t.Optional[bool] = True,
    ):
        """Initialize the pressure profile.

        Unitless arrays are interpreted as pressure in Pa.

        Parameters
        ----------
        array : array-like or astropy.units.Quantity
            Pressure profile in Pa when unitless.

        reverse : t.Optional[bool], optional
            Reverse the pressure profile, by default False

        is_central : t.Optional[bool], optional
            If the pressure profile is central or not, by default True
            **New in v3.2.0**

        """
        array = convert_to_unit_value(array, u.Pa, default_unit=u.Pa)
        array = np.asarray(array, dtype=get_float_dtype())
        super().__init__(self.__class__.__name__, array.shape[-1])

        self.is_central = is_central

        if reverse:
            self.pressure_profile = array[::-1]
        else:
            self.pressure_profile = array

    def compute_pressure_profile(self) -> None:
        """Sets up the pressure profile for the atmosphere model."""
        if self.is_central:
            logp = np.log10(self.pressure_profile)
            gradp = np.gradient(logp)

            self.pressure_profile_levels = 10 ** np.append(
                logp - gradp / 2, logp[-1] + gradp[-1] / 2
            )
        else:
            self.pressure_profile_levels = self.pressure_profile

            self.pressure_profile = self.pressure_profile_levels[:-1] * np.sqrt(
                self.pressure_profile_levels[1:] / self.pressure_profile_levels[:-1]
            )

    @property
    def profile(self) -> npt.NDArray[np.float64]:
        """Pressure profile array."""
        return self.pressure_profile

    def write(self, output: OutputGroup) -> OutputGroup:
        """Write pressure profile to output."""
        pressure = super().write(output)

        return pressure

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Input keywords for pressure profile."""
        return (
            "array",
            "fromarray",
        )

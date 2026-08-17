"""Temperature profile loaded from array."""

import typing as t

import numpy as np
import numpy.typing as npt
from astropy import units as u
from scipy.interpolate import interp1d

from taurex.output import OutputGroup
from taurex.types import get_float_dtype
from taurex.util import convert_to_unit_value

from .tprofile import TemperatureProfile


class TemperatureArray(TemperatureProfile):
    """Temperature profile loaded from array."""

    def __init__(
        self,
        tp_array: t.Optional[t.Union[npt.ArrayLike, u.Quantity]] = None,
        p_points: t.Optional[t.Union[npt.ArrayLike, u.Quantity]] = None,
        reverse: t.Optional[bool] = False,
    ):
        """Initialize the temperature profile.

        Parameters
        ----------
        tp_array:
            Temperature profile array in K when unitless, by default None.
        p_points:
            Pressure points in Pa when unitless, by default None.
        reverse : t.Optional[bool], optional
            Reverse the temperature profile, by default False

        """
        super().__init__(self.__class__.__name__)

        tp_array = convert_to_unit_value(
            tp_array, u.K, default_unit=u.K, equivalencies=u.temperature()
        )
        self._tp_profile = np.asarray(tp_array, dtype=get_float_dtype())
        if reverse:
            self._tp_profile = self._tp_profile[::-1]
        if p_points is not None:
            p_points = convert_to_unit_value(
                p_points, u.Pa, default_unit=u.Pa
            )
            self._p_profile = np.asarray(p_points, dtype=get_float_dtype())
            if reverse:
                self._p_profile = self._p_profile[::-1]
            self._func = interp1d(
                np.log10(self._p_profile),
                self._tp_profile,
                bounds_error=False,
                fill_value=(self._tp_profile[-1], self._tp_profile[0]),
            )
        else:
            self._p_profile = None

    @property
    def profile(self) -> npt.NDArray[np.float64]:
        """Returns temperature profile.

        Returns
        -------
        t_profile:
            temperature profile
        """
        if self._p_profile is None:
            if self._tp_profile.shape[0] == self.nlayers:
                return self._tp_profile
            interp_temp = np.linspace(1.0, 0.0, self._tp_profile.shape[0])
            interp_array = np.linspace(1.0, 0.0, self.nlayers)
            return np.interp(
                interp_array[::-1], interp_temp[::-1], self._tp_profile[::-1]
            )
        else:
            interp_array = np.log10(self.pressure_profile)
            return self._func(interp_array)

    def write(self, output: OutputGroup) -> OutputGroup:
        """Write temperature profile to output."""
        temperature = super().write(output)

        temperature.write_scalar("tp_array", self._tp_profile)

        return temperature

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Return all input keywords."""
        return (
            "array",
            "fromarray",
        )

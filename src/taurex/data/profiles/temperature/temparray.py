"""Temperature profile loaded from array."""
import typing as t

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d

from taurex.output import OutputGroup

from .tprofile import TemperatureProfile


class TemperatureArray(TemperatureProfile):
    """Temperature profile loaded from array."""

    def __init__(
        self,
        tp_array: t.Optional[npt.ArrayLike] = None,
        p_points: t.Optional[npt.ArrayLike] = None,
        reverse: t.Optional[bool] = False,
    ):
        """Initialize the temperature profile.

        Parameters
        ----------
        tp_array:
            Temperature profile array, by default None
        p_points:
            Pressure profile array, by default None
        reverse : t.Optional[bool], optional
            Reverse the temperature profile, by default False

        """
        super().__init__(self.__class__.__name__)

        self._tp_profile = np.array(tp_array)
        if reverse:
            self._tp_profile = self._tp_profile[::-1]
        if p_points is not None:
            self._p_profile = np.array(p_points)
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

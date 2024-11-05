"""Base and Concrete pressure profiles."""

import math
import typing as t

import numpy as np
import numpy.typing as npt

from taurex.data.citation import Citable
from taurex.data.fittable import Fittable, fitparam
from taurex.log import Logger
from taurex.output import OutputGroup
from taurex.output.writeable import Writeable


class PressureProfile(Fittable, Logger, Writeable, Citable):
    """Base pressure class.

    *Abstract Class*

    Simple. Defines the layering
    of the atmosphere. Only requires
    implementation of:

    - :func:`compute_pressure_profile`
    - :func:`profile`

    """

    def __init__(self, name: str, nlayers: int) -> None:
        """Initialize pressure profile.



        Parameters
        ----------

        name: str
            Name used in logging

        nlayers: int
            Number of layers in atmosphere

        """
        Fittable.__init__(self)
        Logger.__init__(self, name)
        self.pressure_profile_levels: npt.NDArray[np.float64] = None
        if nlayers <= 0:
            self.error("Number of layers: [%s] should be greater than 0", nlayers)
            raise ValueError("Number of layers should be at least 1")

        self._nlayers = int(nlayers)

    @property
    def nLayers(self) -> int:  # noqa: N802
        """Number of central layers.

        Returns
        -------
        int
        """

        return self._nlayers

    @property
    def nLevels(self) -> int:  # noqa: N802
        """Number of levels (interface between layers)."""
        return self.nLayers + 1

    def compute_pressure_profile(self) -> None:
        """Compute pressure profile in Pa.


        **Requires implementation**

        Compute pressure profile and
        generate pressure array in Pa

        Returns
        -------
        pressure_profile: :obj:`array`
            Pressure profile array in Pa

        """
        raise NotImplementedError

    @property
    def profile(self) -> npt.NDArray[np.float64]:
        """Pressure at each atmospheric layer (Pascal)

        Returns
        -------
        pressure_profile : :obj:`array`
            Pressure profile array in Pa
        """
        raise NotImplementedError

    def write(self, output: OutputGroup) -> OutputGroup:
        """Write pressure profile to output."""
        pressure = output.create_group("Pressure")
        pressure.write_string("pressure_type", self.__class__.__name__)
        pressure.write_scalar("nlayers", self._nlayers)
        pressure.write_array("profile", self.profile)
        return pressure


class SimplePressureProfile(PressureProfile):
    """A basic pressure profile."""

    WARN = True

    def __init__(
        self,
        nlayers: t.Optional[int] = 100,
        atm_min_pressure: t.Optional[float] = 1e-4,
        atm_max_pressure: t.Optional[float] = 1e6,
    ):
        """Initialize pressure profile.

        Parameters
        ----------
        nlayers : int
            Number of layers in atmosphere

        atm_min_pressure : float
            minimum pressure in Pascal (top of atmosphere)

        atm_max_pressure : float
            maximum pressure in Pascal (surface of planet)

        """
        from warnings import warn

        super().__init__("pressure_profile", nlayers)
        self.pressure_profile = None
        if self.WARN:
            warn(
                "SimplePressureProfile is deprecated. "
                "Use LogPressureProfile instead",
                DeprecationWarning,
            )
        if atm_max_pressure <= atm_min_pressure:
            self.error(
                "Max pressure %1.2e should be greater " "than min pressure %1.2e",
                atm_max_pressure,
                atm_min_pressure,
            )
            raise ValueError("Max pressure is less than minimum pressure")

        self._atm_min_pressure = atm_min_pressure
        self._atm_max_pressure = atm_max_pressure

    def compute_pressure_profile(self) -> None:
        """Set up the pressure profile for the atmosphere model."""

        # set pressure profile of layer boundaries
        # press_exp = np.linspace(np.log(self._atm_min_pressure),
        #                       np.log(self._atm_max_pressure),
        #                       self.nLevels)
        # self.pressure_profile_levels = np.exp(press_exp)[::-1]
        self.pressure_profile_levels = np.logspace(
            math.log10(self._atm_min_pressure),
            math.log10(self._atm_max_pressure),
            self.nLevels,
        )[::-1]
        # get mid point pressure between levels (i.e. get layer pressure)
        # computing geometric
        # average between pressure at n and n+1 level
        self.pressure_profile = self.pressure_profile_levels[:-1] * np.sqrt(
            self.pressure_profile_levels[1:] / self.pressure_profile_levels[:-1]
        )

    @fitparam(
        param_name="atm_min_pressure",
        param_latex=r"$P_\mathrm{min}$",
        default_mode="log",
        default_fit=False,
        default_bounds=[0.1, 1.0],
    )
    def minAtmospherePressure(self) -> float:  # noqa: N802
        """Minimum pressure of atmosphere (top layer) in Pascal"""
        return self._atm_min_pressure

    @minAtmospherePressure.setter
    def minAtmospherePressure(self, value: float) -> None:  # noqa: N802
        self._atm_min_pressure = value

    @fitparam(
        param_name="atm_max_pressure",
        param_latex=r"$P_\mathrm{max}$",
        default_mode="log",
        default_fit=False,
        default_bounds=[0.1, 1.0],
    )
    def maxAtmospherePressure(self) -> float:  # noqa: N802
        """Maximum pressure of atmosphere (surface) in Pascal."""
        return self._atm_max_pressure

    @maxAtmospherePressure.setter
    def maxAtmospherePressure(self, value: float):  # noqa: N802
        """Set the maximum pressure of the atmosphere (surface) in Pascal"""
        self._atm_max_pressure = value

    @property
    def profile(self) -> npt.NDArray[np.float64]:
        """Pressure at each atmospheric layer (Pascal)"""
        return self.pressure_profile

    def write(self, output: OutputGroup) -> OutputGroup:
        """Write pressure profile to output."""
        pressure = super().write(output)

        pressure.write_scalar("atm_max_pressure", self._atm_max_pressure)
        pressure.write_scalar("atm_min_pressure", self._atm_min_pressure)

        return pressure

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        return (
            "simple",
            "hydrostatic",
            "logpressure",
        )


class LogPressureProfile(SimplePressureProfile):
    """A pressure profile built from a logspace."""

    WARN = False
    pass

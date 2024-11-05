"""Module for NPoint temperature profile."""

import typing as t

import numpy as np
import numpy.typing as npt

from taurex.data.fittable import fitparam
from taurex.exceptions import InvalidModelException
from taurex.output import OutputGroup
from taurex.util import movingaverage

from .tprofile import TemperatureProfile


class InvalidTemperatureException(InvalidModelException):
    """Exception that is called when temperature profile is invalid."""

    pass


class NPoint(TemperatureProfile):
    """Temperature profile defined and smoothed by user points.

    A temperature profile that is defined at various heights of the
    atmopshere and then smoothend.

    At minimum, temepratures on both the top ``T_top`` and surface
    ``T_surface`` must be defined.
    If any intermediate points are given as ``temperature_points``
    then the same number of ``pressure_points``
    must be given as well.

    A 2-point temperature profile has ``len(temperature_points) == 0``
    A 3-point temperature profile has ``len(temperature_points) == 1``

    etc.
    """

    def __init__(
        self,
        T_surface: t.Optional[float] = 1500.0,  # noqa: N803
        T_top: t.Optional[float] = 200.0,  # noqa: N803
        P_surface: t.Optional[float] = None,  # noqa: N803
        P_top: t.Optional[float] = None,  # noqa: N803
        temperature_points: t.Optional[npt.ArrayLike] = None,
        pressure_points: t.Optional[npt.ArrayLike] = None,
        smoothing_window: t.Optional[int] = 10,
        limit_slope: t.Optional[int] = 9999999,
    ):
        """Initialize NPoint temperature profile.

        Parameters
        ----------
        T_surface : float
            BOA temperature in Kelvin
        T_top : float
            TOA temperature in Kelvin
        P_surface : float
            BOA pressure in Pa
        P_top : float
            TOA pressure in Pa
        temperature_points : array-like
            Temperature points
        pressure_points : array-like
            Pressure points
        smoothing_window : int
            Smoothing window
        limit_slope : int
            Gradient limit to be considered valid
        """
        temperature_points = (
            temperature_points if temperature_points is not None else []
        )
        pressure_points = pressure_points if pressure_points is not None else []

        super().__init__(f"{len(temperature_points) + 2}Point")

        if not hasattr(temperature_points, "__len__"):
            raise ValueError("t_point is not an iterable")

        if len(temperature_points) != len(pressure_points):
            self.error("Number of temeprature points != number of " "pressure points")
            self.error(
                "len(t_points) = %s /= " "len(p_points) = %s",
                len(temperature_points),
                len(pressure_points),
            )
            raise ValueError("Incorrect_number of temp and pressure points")

        self.info("Npoint temeprature profile is initialized")
        self.debug("Passed temeprature points %s", temperature_points)
        self.debug("Passed pressure points %s", pressure_points)
        self._t_points = temperature_points
        self._p_points = pressure_points
        self._T_surface = T_surface
        self._T_top = T_top
        self._P_surface = P_surface
        self._P_top = P_top
        self._smooth_window = smoothing_window
        self._limit_slope = limit_slope
        self.generate_pressure_fitting_params()
        self.generate_temperature_fitting_params()

    @fitparam(
        param_name="T_surface",
        param_latex=r"$T_\mathrm{surf}$",
        default_fit=False,
        default_bounds=[300, 2500],
    )
    def temperatureSurface(self) -> float:  # noqa: N802
        """Temperature at planet surface in Kelvin"""
        return self._T_surface

    @temperatureSurface.setter
    def temperatureSurface(self, value: float) -> None:  # noqa: N802
        """Temperature at planet surface in Kelvin"""
        self._T_surface = value

    @fitparam(
        param_name="T_top",
        param_latex=r"$T_\mathrm{top}$",
        default_fit=False,
        default_bounds=[300, 2500],
    )
    def temperatureTop(self) -> float:  # noqa: N802
        """Temperature at top of atmosphere in Kelvin"""
        return self._T_top

    @temperatureTop.setter
    def temperatureTop(self, value: float) -> None:  # noqa: N802
        """Temperature at top of atmosphere in Kelvin"""
        self._T_top = value

    @fitparam(
        param_name="P_surface",
        param_latex=r"$P_\mathrm{surf}$",
        default_fit=False,
        default_bounds=[1e3, 1e2],
        default_mode="log",
    )
    def pressureSurface(self) -> float:  # noqa: N802
        """Pressure at planet surface in Pa"""
        return self._P_surface

    @pressureSurface.setter
    def pressureSurface(self, value: float) -> None:  # noqa: N802
        """Pressure at planet surface in Pa"""
        self._P_surface = value

    @fitparam(
        param_name="P_top",
        param_latex=r"$P_\mathrm{top}$",
        default_fit=False,
        default_bounds=[1e-5, 1e-4],
        default_mode="log",
    )
    def pressureTop(self) -> float:  # noqa: N802
        """Pressure at top of atmosphere in Pa"""
        return self._P_top

    @pressureTop.setter
    def pressureTop(self, value: float) -> None:  # noqa: N802
        """Pressure at top of atmosphere in Pa"""
        self._P_top = value

    def generate_pressure_fitting_params(self) -> None:
        """Generates the fitting parameters for the pressure points.

        These are given the name ``P_point(number)`` for example, if two extra
        pressure points are defined between the top and surface then the
        fitting parameters generated are ``P_point0`` and ``P_point1``
        """

        bounds = [1e5, 1e3]
        for idx, _ in enumerate(self._p_points):
            point_num = idx + 1
            param_name = f"P_point{point_num}"
            param_latex = f"$P_{point_num}$"

            def read_point(self, idx=idx):
                return self._p_points[idx]

            def write_point(self, value, idx=idx):
                self._p_points[idx] = value

            read_point.__doc__ = f"Pressure point {point_num} in Pa"

            fget_point = read_point
            fset_point = write_point
            self.debug("FGet_location %s", fget_point)
            default_fit = False
            self.add_fittable_param(
                param_name,
                param_latex,
                fget_point,
                fset_point,
                "log",
                default_fit,
                bounds,
            )

    def generate_temperature_fitting_params(self) -> None:
        """Generates the fitting parameters for the temperature points.

        These are given the name ``T_point(number)`` for example, if two extra
        temeprature points are defined between the top and surface then the
        fitting parameters generated are ``T_point0`` and ``T_point1``
        """

        bounds = [300, 2500]
        for idx, _ in enumerate(self._t_points):
            point_num = idx + 1
            param_name = f"T_point{point_num}"
            param_latex = f"$T_{point_num}$"

            def read_point(self, idx=idx):
                return self._t_points[idx]

            def write_point(self, value, idx=idx):
                self._t_points[idx] = value

            read_point.__doc__ = f"Temperature point {point_num} in K"

            fget_point = read_point
            fset_point = write_point
            self.debug("FGet_location %s %s", fget_point, fget_point(self))
            default_fit = False
            self.add_fittable_param(
                param_name,
                param_latex,
                fget_point,
                fset_point,
                "linear",
                default_fit,
                bounds,
            )

    def check_profile(self, ppt: npt.ArrayLike, tpt: npt.ArrayLike):
        """Checks the validity of the temperature profile.

        Parameters
        ----------
        ppt : array-like
            Pressure points
        tpt : array-like
            Temperature points

        Raises
        ------
        InvalidTemperatureException
            If the temperature profile is invalid


        """
        if any(ppt[i] <= ppt[i + 1] for i in range(len(ppt) - 1)):
            self.warning(
                "Temperature profile is not valid - a pressure point is inverted"
            )
            raise InvalidTemperatureException

        if any(
            abs((tpt[i + 1] - tpt[i]) / (np.log10(ppt[i + 1]) - np.log10(ppt[i])))
            >= self._limit_slope
            for i in range(len(ppt) - 1)
        ):
            self.warning("Temperature profile is not valid - profile slope too high")
            raise InvalidTemperatureException

    @property
    def profile(self) -> npt.NDArray[np.float64]:
        """Returns a smoothed temperature profile.

        Computes the temperature profile from the given points and then
        smooths it using a moving average.

        """
        t_nodes = [self._T_surface, *self._t_points, self._T_top]

        p_surface = self._P_surface
        if p_surface is None or p_surface < 0:
            p_surface = self.pressure_profile[0]

        p_top = self._P_top
        if p_top is None or p_top < 0:
            p_top = self.pressure_profile[-1]

        p_nodes = [p_surface, *self._p_points, p_top]

        self.check_profile(p_nodes, t_nodes)

        smooth_window = self._smooth_window

        if np.all(t_nodes == t_nodes[0]):
            return np.ones_like(self.pressure_profile) * t_nodes[0]

        tp = np.interp(
            (np.log10(self.pressure_profile[::-1])),
            np.log10(p_nodes[::-1]),
            t_nodes[::-1],
        )

        # smoothing T-P profile
        wsize = int(self.nlayers * (smooth_window / 100.0))
        if wsize % 2 == 0:
            wsize += 1
        tp_smooth = movingaverage(tp, wsize)
        border = np.int64((len(tp) - len(tp_smooth)) / 2)

        foo = tp[::-1]
        if len(tp_smooth) == len(foo):
            foo = tp_smooth[::-1]
        else:
            foo[border:-border] = tp_smooth[::-1]

        return foo

    def write(self, output: OutputGroup) -> OutputGroup:
        """Write NPoint temperature profile to output group."""
        temperature = super().write(output)

        temperature.write_scalar("T_surface", self._T_surface)
        temperature.write_scalar("T_top", self._T_top)
        temperature.write_array("temperature_points", np.array(self._t_points))

        p_surface = self._P_surface
        p_top = self._P_top
        if not p_surface:
            p_surface = -1
        if not p_top:
            p_top = -1

        temperature.write_scalar("P_surface", p_surface)
        temperature.write_scalar("P_top", p_top)
        temperature.write_array("pressure_points", np.array(self._p_points))

        temperature.write_scalar("smoothing_window", self._smooth_window)

        return temperature

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Return all input keywords."""
        return ("npoint",)

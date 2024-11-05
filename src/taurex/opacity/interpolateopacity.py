import typing as t

import numpy as np
import numpy.typing as npt

from taurex.util.math import (
    intepr_bilin,
    interp_exp_and_lin,
    interp_exp_only,
    interp_lin_only,
)

from .opacity import Opacity

InterpModeType = t.Literal["linear", "exp"]


class InterpolatingOpacity(Opacity):
    """Provides interpolation methods."""

    def __init__(
        self, name: str, interpolation_mode: t.Optional[InterpModeType] = "linear"
    ) -> None:
        """Initialize interpolating opacity.

        Parameters
        ----------
        name: str
            Name of opacity for logging.
        interpolation_mode: str
            Interpolation mode. Either 'linear' or 'exp'.


        """
        super().__init__(name)
        self._interp_mode = interpolation_mode

    @property
    def pressureMax(self) -> float:  # noqa: N802
        """Maximum pressure in opacity grid."""
        return self.pressureGrid[-1]

    @property
    def pressureMin(self) -> float:  # noqa: N802
        """Minimum pressure in opacity grid."""
        return self.pressureGrid[0]

    @property
    def temperatureMax(self) -> float:  # noqa: N802
        """Maximum temperature in opacity grid."""
        return self.temperatureGrid[-1]

    @property
    def temperatureMin(self) -> float:  # noqa: N802
        """Minimum temperature in opacity grid."""
        return self.temperatureGrid[0]

    @property
    def xsecGrid(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Opacity grid."""
        raise NotImplementedError

    @property
    def logPressure(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Log pressure grid."""
        return np.log10(self.pressureGrid)

    @property
    def pressureBounds(self) -> t.Tuple[float, float]:  # noqa: N802
        """Pressure bounds."""
        return self.logPressure.min(), self.logPressure.max()

    @property
    def temperatureBounds(self) -> t.Tuple[float, float]:  # noqa: N802
        """Temperature bounds."""
        return self.temperatureGrid.min(), self.temperatureGrid.max()

    def find_closest_index(
        self, temperature: float, pressure: float
    ) -> t.Tuple[int, int, int, int]:
        """Find closest pair index to temperature and pressure.

        Closest pair index is the index on either side of
        temperature and pressure.


        """
        from taurex.util import find_closest_pair

        # t_min = self.temperatureGrid.searchsorted(T, side='right')-1
        # t_min = max(0, t_min)
        # t_max = t_min+1
        # t_max = min(len(self.temperatureGrid)-1, t_max)
        # p_min = self.pressureGrid.searchsorted(P, side='right')-1
        # p_min = max(0, p_min)
        # p_max = p_min+1
        # p_max = min(len(self.pressureGrid)-1, p_max)

        t_min, t_max = find_closest_pair(self.temperatureGrid, temperature)
        p_min, p_max = find_closest_pair(self.logPressure, pressure)

        return t_min, t_max, p_min, p_max

    def set_interpolation_mode(self, interp_mode: InterpModeType) -> None:
        """Set interpolation mode."""
        self._interp_mode = interp_mode.strip()

    def interp_temp_only(
        self,
        temperature: float,
        temp_idx_min: int,
        temp_idx_max: int,
        pressure_idx: int,
        filt: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.float64]:
        """Interpolate temperature only."""
        temperature_max = self.temperatureGrid[temp_idx_max]
        temperature_min = self.temperatureGrid[temp_idx_min]
        fx0 = self.xsecGrid[pressure_idx, temp_idx_min, filt]
        fx1 = self.xsecGrid[pressure_idx, temp_idx_max, filt]

        if self._interp_mode == "linear":
            return interp_lin_only(
                fx0, fx1, temperature, temperature_min, temperature_max
            )
        elif self._interp_mode == "exp":
            return interp_exp_only(
                fx0, fx1, temperature, temperature_min, temperature_max
            )
        else:
            raise ValueError(f"Unknown interpolation mode {self._interp_mode}")

    def interp_pressure_only(
        self,
        pressure: float,
        p_idx_min: int,
        p_idx_max: int,
        temperature_idx: int,
        filt: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.float64]:
        """Interpolate pressure only."""
        pressure_max = self.logPressure[p_idx_max]
        pressure_min = self.logPressure[p_idx_min]
        fx0 = self.xsecGrid[p_idx_min, temperature_idx, filt]
        fx1 = self.xsecGrid[p_idx_max, temperature_idx, filt]

        return interp_lin_only(fx0, fx1, pressure, pressure_min, pressure_max)

    def interp_bilinear_grid(
        self,
        temperature: float,
        pressure: float,
        t_idx_min: int,
        t_idx_max: int,
        p_idx_min: int,
        p_idx_max: int,
        wngrid_filter: npt.NDArray[np.bool_] = None,
    ) -> npt.NDArray[np.float64]:
        """Interpolate both temperature and pressure."""
        self.debug(
            "Interpolating %s %s %s %s %s %s",
            temperature,
            pressure,
            t_idx_min,
            t_idx_max,
            p_idx_min,
            p_idx_max,
        )

        check_pressure_max = pressure >= self.pressureMax
        check_temperature_max = temperature >= self.temperatureMax

        min_pressure, max_pressure = self.pressureBounds
        min_temperature, max_temperature = self.temperatureBounds

        check_pressure_max = pressure >= max_pressure
        check_temperature_max = temperature >= max_temperature

        check_pressure_min = pressure < min_pressure
        check_temperature_min = temperature < min_temperature

        self.debug(
            "Check pressure min/max %s/%s", check_pressure_min, check_pressure_max
        )
        self.debug(
            "Check temperature min/max %s/%s",
            check_temperature_min,
            check_temperature_max,
        )
        # Are we both max?
        if check_pressure_max and check_temperature_max:
            self.debug("Maximum Temperature pressure reached. Using last")
            return self.xsecGrid[-1, -1, wngrid_filter].ravel()

        if check_pressure_min and check_temperature_min:
            return np.zeros_like(self.xsecGrid[0, 0, wngrid_filter]).ravel()

        # Max pressure
        if check_pressure_max:
            self.debug("Max pressure reached. Interpolating temperature only")
            return self.interp_temp_only(
                temperature, t_idx_min, t_idx_max, -1, wngrid_filter
            )

        # Max temperature
        if check_temperature_max:
            self.debug("Max temperature reached. Interpolating pressure only")
            return self.interp_pressure_only(
                pressure, p_idx_min, p_idx_max, -1, wngrid_filter
            )

        if check_pressure_min:
            self.debug("Min pressure reached. Interpolating temperature only")
            return self.interp_temp_only(
                temperature, t_idx_min, t_idx_max, 0, wngrid_filter
            ).ravel()

        if check_temperature_min:
            self.debug("Min temperature reached. Interpolating pressure only")
            return self.interp_pressure_only(
                pressure, p_idx_min, p_idx_max, 0, wngrid_filter
            ).ravel()

        q_11 = self.xsecGrid[p_idx_min, t_idx_min][wngrid_filter].ravel()
        q_12 = self.xsecGrid[p_idx_min, t_idx_max][wngrid_filter].ravel()
        q_21 = self.xsecGrid[p_idx_max, t_idx_min][wngrid_filter].ravel()
        q_22 = self.xsecGrid[p_idx_max, t_idx_max][wngrid_filter].ravel()

        temperature_max = self.temperatureGrid[t_idx_max]
        temperature_min = self.temperatureGrid[t_idx_min]
        pressure_max = self.logPressure[p_idx_max]
        pressure_min = self.logPressure[p_idx_min]

        if self._interp_mode == "linear":
            return intepr_bilin(
                q_11,
                q_12,
                q_21,
                q_22,
                temperature,
                temperature_min,
                temperature_max,
                pressure,
                pressure_min,
                pressure_max,
            )
        elif self._interp_mode == "exp":
            return interp_exp_and_lin(
                q_11,
                q_12,
                q_21,
                q_22,
                temperature,
                temperature_min,
                temperature_max,
                pressure,
                pressure_min,
                pressure_max,
            )
        else:
            raise ValueError(f"Unknown interpolation mode {self._interp_mode}")

    def compute_opacity(
        self,
        temperature: float,
        pressure: float,
        wngrid: t.Optional[npt.NDArray[np.float64]] = None,
    ) -> npt.NDArray[np.float64]:
        """Compute opacity.

        Interpolates the opacity grid to the given temperature and pressure.

        Parameters
        ----------
        temperature: float
            Temperature in K
        pressure: float
            Pressure in Pa
        wngrid: :obj:`array`, optional
            Wavenumber grid to restrict to.

        Returns
        -------
        :obj:`array`
            Interpolated opacity.

        """
        import math

        logpressure = math.log10(pressure)
        return (
            self.interp_bilinear_grid(
                temperature,
                logpressure,
                *self.find_closest_index(temperature, logpressure),
                wngrid,
            )
            / 10000
        )

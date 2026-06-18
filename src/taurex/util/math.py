"""Optimized Math functions used in taurex."""

import typing as t

import numpy as np
import numpy.typing as npt

from taurex.log import setup_log
from taurex.types import AnyValType


_log = setup_log(__name__)

try:
    from .math_numba import intepr_bilin_numba_II
    from .math_numba import interp_lin_numba

    numba_enabled = True
except ImportError:
    _log.warning("Numba not installed, using numpy")
    numba_enabled = False


def interp_exp_and_lin_numpy(
    x11: npt.NDArray[np.float64],
    x12: npt.NDArray[np.float64],
    x21: npt.NDArray[np.float64],
    x22: npt.NDArray[np.float64],
    temperature: float,
    temperature_min: float,
    temperature_max: float,
    pressure: float,
    pressure_min: float,
    pressure_max: float,
) -> npt.NDArray[np.float64]:
    """2D interpolation.

    Applies linear interpolation across pressure and e interpolation
    across temperature between pressure_min,pressure_max and
    temperature_min,temperature_max

    Parameters
    ----------
    x11: array
        Array corresponding to pressure_min,temperature_min

    x12: array
        Array corresponding to pressure_min,temperature_max

    x21: array
        Array corresponding to pressure_max,temperature_min

    x22: array
        Array corresponding to pressure_max,temperature_max

    temperature: float
        Coordinate to exp interpolate to

    temperature_min: float
        Nearest known temperature coordinate where
        temperature_min < temperature

    temperature_max: float
        Nearest known temperature coordinate where
        temperature < temperature_max

    pressure: float
        Coordinate to linear interpolate to

    pressure_min: float
        Nearest known pressure coordinate where
        pressure_min < pressure

    pressure_max: float
        Nearest known pressure coordinate where
        pressure < pressure_max

    """
    return (
        (x11 * (pressure_max - pressure_min) - (pressure - pressure_min) * (x11 - x21))
        * np.exp(
            temperature_max
            * (-temperature + temperature_min)
            * np.log(
                (
                    x11 * (pressure_max - pressure_min)
                    - (pressure - pressure_min) * (x11 - x21)
                )
                / (
                    x12 * (pressure_max - pressure_min)
                    - (pressure - pressure_min) * (x12 - x22)
                )
            )
            / (temperature * (temperature_max - temperature_min))
        )
        / (pressure_max - pressure_min)
    )


def interp_exp_numpy(
    x11: npt.NDArray[np.float64],
    x12: npt.NDArray[np.float64],
    temperature,
    temperature_min,
    temperature_max,
) -> npt.NDArray[np.float64]:
    """Exponential interpolation.

    Parameters
    ----------
    x11: array
        Array corresponding to temperature_min

    x12: array
        Array corresponding to temperature_max

    temperature: float
        Coordinate to exp interpolate to

    temperature_min: float
        Nearest known temperature coordinate where
        temperature_min < temperature

    temperature_max: float
        Nearest known temperature coordinate where
        temperature < temperature_max

    Returns
    -------
    array
        Interpolated array.

    """
    return x11 * np.exp(
        temperature_max
        * (-temperature + temperature_min)
        * np.log(x11 / x12)
        / (temperature * (temperature_max - temperature_min))
    )


def interp_lin_numpy(
    x11: npt.NDArray[np.float64],
    x12: npt.NDArray[np.float64],
    pressure: float,
    pressure_min: float,
    pressure_max: float,
) -> npt.NDArray[np.float64]:
    """Linear interpolation.

    Parameters
    ----------
    x11: array
        Array corresponding to pressure_min

    x12: array
        Array corresponding to pressure_max

    pressure: float
        Coordinate to linear interpolate to

    pressure_min: float
        Nearest known pressure coordinate where
        pressure_min < pressure

    pressure_max: float
        Nearest known pressure coordinate where
        pressure < pressure_max

    Returns
    -------
    array
        Interpolated array.

    """
    return (
        x11 * (pressure_max - pressure_min) - (pressure - pressure_min) * (x11 - x12)
    ) / (pressure_max - pressure_min)


def interp_bilin_numpy(
    x11: npt.NDArray[np.float64],
    x12: npt.NDArray[np.float64],
    x21: npt.NDArray[np.float64],
    x22: npt.NDArray[np.float64],
    temperature: float,
    temperature_min: float,
    temperature_max: float,
    pressure: float,
    pressure_min: float,
    pressure_max: float,
) -> npt.NDArray[np.float64]:
    """Bilinear interpolation.

    Parameters
    ----------
    x11: array
        Array corresponding to pressure_min,temperature_min

    x12: array
        Array corresponding to pressure_min,temperature_max

    x21: array
        Array corresponding to pressure_max,temperature_min

    x22: array
        Array corresponding to pressure_max,temperature_max

    temperature: float
        Temperature coordinate

    temperature_min: float
        Minimum temperature coordinate

    temperature_max: float
        Maximum temperature coordinate

    pressure: float
        Pressure coordinate

    pressure_min: float
        Minimum pressure coordinate

    pressure_max: float
        Maximum pressure coordinate

    Returns
    -------
    array
        Interpolated array.

    """
    pressure_diff = pressure_max - pressure_min
    temperature_diff = temperature_max - temperature_min
    pressure_scale = (pressure - pressure_min) / pressure_diff
    temperature_scale = (temperature - temperature_min) / temperature_diff

    return (
        x11
        - pressure_scale * (x11 - x21)
        - pressure_scale * temperature_scale * (x21 - x11 + x12 - x22)
        - temperature_scale * (x11 - x12)
    )


def intepr_bilin_numexpr(
    x11: npt.NDArray[np.float64],
    x12: npt.NDArray[np.float64],
    x21: npt.NDArray[np.float64],
    x22: npt.NDArray[np.float64],
    temperature: float,
    temperature_min: float,
    temperature_max: float,
    pressure: float,
    pressure_min: float,
    pressure_max: float,
) -> npt.NDArray[np.float64]:
    """Bilinear interpolation using numexpr.

    Parameters
    ----------
    x11: array
        Array corresponding to pressure_min,temperature_min

    x12: array
        Array corresponding to pressure_min,temperature_max

    x21: array
        Array corresponding to pressure_max,temperature_min

    x22: array
        Array corresponding to pressure_max,temperature_max

    temperature: float
        Temperature coordinate

    temperature_min: float
        Minimum temperature coordinate

    temperature_max: float
        Maximum temperature coordinate

    pressure: float
        Pressure coordinate

    pressure_min: float
        Minimum pressure coordinate

    pressure_max: float
        Maximum pressure coordinate

    Returns
    -------
    array
        Interpolated array.

    """
    import numexpr as ne

    return ne.evaluate(
        "(x11*(pressure_max - pressure_min)*"
        "(temperature_max - temperature_min)"
        " - (pressure - pressure_min)*(temperature_max - "
        "temperature_min)*(x11 - x21)"
        " - (temperature - temperature_min)*"
        "(-(pressure - pressure_min)*(x11 - x21)"
        " + (pressure - pressure_min)*(x12 - x22) + "
        "(pressure_max - pressure_min)*(x11 - x12)))"
        "/((pressure_max - pressure_min)*"
        "(temperature_max - temperature_min))"
    )


def intepr_bilin_double(
    x11: npt.NDArray[np.float64],
    x12: npt.NDArray[np.float64],
    x21: npt.NDArray[np.float64],
    x22: npt.NDArray[np.float64],
    temperature: float,
    temperature_min: float,
    temperature_max: float,
    pressure: float,
    pressure_min: float,
    pressure_max: float,
) -> npt.NDArray[np.float64]:
    """Bilinear interpolation using double.

    Parameters
    ----------
    x11: array
        Array corresponding to pressure_min,temperature_min

    x12: array
        Array corresponding to pressure_min,temperature_max

    x21: array
        Array corresponding to pressure_max,temperature_min

    x22: array
        Array corresponding to pressure_max,temperature_max

    temperature: float
        Temperature coordinate

    temperature_min: float
        Minimum temperature coordinate

    temperature_max: float
        Maximum temperature coordinate

    pressure: float
        Pressure coordinate

    pressure_min: float
        Minimum pressure coordinate

    pressure_max: float
        Maximum pressure coordinate

    Returns
    -------
    array
        Interpolated array.

    """
    return interp_lin_only(
        interp_lin_only(
            x11,
            x12,
            temperature,
            temperature_min,
            temperature_max,
        ),
        interp_lin_only(
            x21,
            x22,
            temperature,
            temperature_min,
            temperature_max,
        ),
        pressure,
        pressure_min,
        pressure_max,
    )


def intepr_bilin_old(
    x11: npt.NDArray[np.float64],
    x12: npt.NDArray[np.float64],
    x21: npt.NDArray[np.float64],
    x22: npt.NDArray[np.float64],
    temperature: float,
    temperature_min: float,
    temperature_max: float,
    pressure: float,
    pressure_min: float,
    pressure_max: float,
) -> npt.NDArray[np.float64]:
    """Bilinear interpolation old.

    Parameters
    ----------
    x11: array
        Array corresponding to pressure_min,temperature_min

    x12: array
        Array corresponding to pressure_min,temperature_max

    x21: array
        Array corresponding to pressure_max,temperature_min

    x22: array
        Array corresponding to pressure_max,temperature_max

    temperature: float
        Temperature coordinate

    temperature_min: float
        Minimum temperature coordinate

    temperature_max: float
        Maximum temperature coordinate

    pressure: float
        Pressure coordinate

    pressure_min: float
        Minimum pressure coordinate

    pressure_max: float
        Maximum pressure coordinate

    Returns
    -------
    array
        Interpolated array.

    """
    return (
        x11 * (pressure_max - pressure_min) * (temperature_max - temperature_min)
        - (pressure - pressure_min) * (temperature_max - temperature_min) * (x11 - x21)
        - (temperature - temperature_min)
        * (
            -(pressure - pressure_min) * (x11 - x21)
            + (pressure - pressure_min) * (x12 - x22)
            + (pressure_max - pressure_min) * (x11 - x12)
        )
    ) / ((pressure_max - pressure_min) * (temperature_max - temperature_min))


def compute_rayleigh_cross_section(
    wngrid: npt.NDArray[np.float64],
    n: float,
    n_air: t.Optional[float] = 2.6867805e25,
    king: t.Optional[float] = 1.0,
) -> npt.NDArray[np.float64]:
    """Compute Rayleigh cross section.

    Parameters
    ----------
    wngrid : npt.NDArray[np.float64]
        Wavenumber grid.
    n : float
        Refractive index.
    n_air : t.Optional[float], optional
        Number density of air, by default 2.6867805e25.
    king : t.Optional[float], optional
        King correction factor, by default 1.0.

    Returns
    -------
    npt.NDArray[np.float64]
        Rayleigh cross section.

    """
    wlgrid = (10000 / wngrid) * 1e-6

    n_factor = (n**2 - 1) / (n_air * (n**2 + 2))
    sigma = 24.0 * (np.pi**3) * king * (n_factor**2) / (wlgrid**4)

    return sigma


def test_nan(val: t.Union[float, npt.ArrayLike]) -> bool:
    """Test if a value is nan.

    Parameters
    ----------
    val : t.Union[float, npt.ArrayLike]
        Value to test.

    Returns
    -------
    bool
        True if nan, False otherwise.

    """
    if hasattr(val, "__len__"):
        try:
            return np.isnan(val).any()
        except TypeError:
            # print(type(val))
            return True
    else:
        return val != val


# Choose the best functions for the task
if numba_enabled:
    interp_lin_only = interp_lin_numba
    intepr_bilin = intepr_bilin_numba_II
else:
    interp_lin_only = interp_lin_numpy
    intepr_bilin = interp_bilin_numpy
interp_exp_and_lin = interp_exp_and_lin_numpy
interp_exp_only = interp_exp_numpy


class OnlineVariance:
    """Uses the M2 algorithm to compute the variance in a streaming fashion."""

    def __init__(self) -> None:
        """Initialise the class."""
        self.reset()

    def reset(self) -> None:
        """Reset the class."""
        self.count = 0.0
        self.wcount = 0.0
        self.wcount2 = 0.0
        self.mean = None
        self.M2 = None

    def update(
        self,
        value: AnyValType,
        weight: t.Optional[float] = 1.0,
    ):
        """Update the variance.

        Parameters
        ----------
        value : AnyValType
            Value to update with.
        weight : t.Optional[float], optional
            Weight for the value, by default 1.0.

        """
        self.count += 1
        self.wcount += weight
        self.wcount2 += weight * weight

        if self.mean is None:
            self.mean = value * 0.0
            self.M2 = value * 0.0

        mean_old = self.mean
        try:
            self.mean = mean_old + (weight / self.wcount) * (value - mean_old)
        except ZeroDivisionError:
            self.mean = value * 0.0
        self.M2 += weight * (value - mean_old) * (value - self.mean)

    @property
    def variance(self) -> float:
        """Return the variance."""
        if self.count < 2:
            return np.nan
        else:
            return self.M2 / self.wcount

    @property
    def sampleVariance(self) -> AnyValType:  # noqa: N802
        """Return the sample variance."""
        if self.count < 2:
            return np.nan
        else:
            return self.M2 / (self.wcount - 1)

    def combine_variance(
        self,
        averages: npt.ArrayLike,
        variance: npt.ArrayLike,
        counts: npt.ArrayLike,
    ) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Combine different variance calculations together.

        Parameters
        ----------
        averages : npt.ArrayLike
            Averages.
        variance : npt.ArrayLike
            Variances.
        counts : npt.ArrayLike
            Counts.

        Returns
        -------
        t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
            Combined average and variance.

        """
        average = None
        size = np.sum(counts)
        for avg, cnt in zip(averages, counts, strict=True):
            if cnt == 0:
                continue

            # print('avg',avg)
            if avg is not None and avg is not np.nan:
                if average is None:
                    average = avg * cnt
                else:
                    average += avg * cnt
        average /= size
        # print('AVERGAE',average)
        counts = np.array(counts) * size / np.sum(counts)

        squares = None

        for avg, cnt, var in zip(averages, counts, variance, strict=True):
            # print('COUNT ',cnt)
            if cnt == 0.0:
                continue
            if cnt > 0.0:
                if squares is None:
                    squares = cnt * (average - avg) ** 2
                else:
                    squares += cnt * (average - avg) ** 2
            if var is not np.nan:
                squares += cnt * var
        # squares = counts*variances
        # squares += counts*(average - averages)**2

        return average, squares / size

    def parallelVariance(self) -> AnyValType:  # noqa: N802
        """Compute the variance in parallel."""
        from taurex import mpi

        variance = self.variance

        mean = self.mean
        if mean is None:
            mean = np.nan

        variances = mpi.allgather(variance)

        averages = mpi.allgather(mean)
        counts = mpi.allgather(self.wcount)
        all_counts = mpi.allgather(self.count)

        if sum(all_counts) < 2:
            return np.nan
        else:
            finalvariance = self.combine_variance(averages, variances, counts)
            return finalvariance[-1]

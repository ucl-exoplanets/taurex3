"""Handles how priors are defined for retrievals."""
import enum
import math
import typing as t

import scipy.stats as stats

from taurex.log import Logger


class PriorMode(enum.Enum):
    """Defines the type of prior space."""

    LINEAR = 0
    LOG = 1


class Prior(Logger):
    """Defines a prior function"""

    def __init__(self) -> None:
        """Initialise prior"""
        super().__init__(self.__class__.__name__)
        self._prior_mode = PriorMode.LINEAR

    @property
    def priorMode(self) -> PriorMode:  # noqa: N802
        """Defined prior mode."""
        return self._prior_mode

    def sample(self, x: float) -> float:
        """Sample the prior.

        Parameters
        ----------
        x : float
            A random number between 0 and 1.

        Raises
        ------
        NotImplementedError
            [description]

        """
        raise NotImplementedError

    def prior(self, value: float) -> float:
        """Convert a value from prior space to linear space."""
        if self._prior_mode is PriorMode.LINEAR:
            return value
        else:
            return 10**value

    def params(self) -> str:
        """Return the parameters of the prior in string form.

        Raises:
            NotImplementedError: [description]

        """
        raise NotImplementedError

    def boundaries(self) -> t.Tuple[float, float]:
        """Return the (rough) boundaries of the prior.

        Raises:
            NotImplementedError: [description]

        """
        raise NotImplementedError


class Uniform(Prior):
    """Defines a uniform prior."""

    def __init__(
        self,
        bounds: t.Optional[t.Tuple[float, float]] = (
            0.0,
            1.0,
        ),
    ):
        """Initialise uniform prior.

        Parameters
        ----------
        bounds : tuple, optional
            Bounds of the prior, by default (0.0, 1.0)

        Raises
        ------
        ValueError
            If no bounds are defined.

        """
        super().__init__()
        if bounds is None:
            self.error("No bounds defined")
            raise ValueError("No bounds defined")

        self.set_bounds(bounds)

    def set_bounds(self, bounds: t.Tuple[float, float]):
        """Set the bounds of the prior.

        Parameters
        ----------
        bounds : t.Tuple[float, float]
            Bounds of the prior.

        """
        self._low_bounds = min(*bounds)
        self.debug("Lower bounds: %s", self._low_bounds)
        self._up_bounds = max(*bounds)
        self._scale = self._up_bounds - self._low_bounds
        self.debug("Scale: %s", self._scale)

    def sample(self, x: float) -> float:
        """Sample the prior.

        Parameters
        ----------
        x : float
            A random number between 0 and 1.

        """
        return stats.uniform.ppf(x, loc=self._low_bounds, scale=self._scale)

    def params(self):
        """Return the parameters of the prior in string form.

        Returns
        -------
        str
            String representation of the prior.

        """
        return f"Bounds = [{self._low_bounds},{self._up_bounds}]"

    def boundaries(self) -> t.Tuple[float, float]:
        """Return the boundaries of the prior."""
        return self._low_bounds, self._up_bounds


class LogUniform(Uniform):
    """Defines a log-uniform prior."""

    def __init__(
        self,
        bounds: t.Optional[t.Tuple[float, float]] = (
            0.0,
            1.0,
        ),
        lin_bounds=None,
    ):
        """Initialise log-uniform prior.

        Parameters
        ----------
        bounds : list, optional
            Bounds of the prior in logspace, by default [0.0, 1.0]

        lin_bounds : list, optional
            Bounds of the prior in linear space, by default None

        """
        if lin_bounds is not None:
            bounds = [math.log10(x) for x in lin_bounds]
        super().__init__(bounds=bounds)
        self._prior_mode = PriorMode.LOG


class Gaussian(Prior):
    """Defines a gaussian prior."""

    def __init__(self, mean: t.Optional[float] = 0.5, std: t.Optional[float] = 0.25):
        """Initialise gaussian prior.

        Parameters
        ----------
        mean : float, optional
            Mean of the gaussian, by default 0.5
        std : float, optional
            Standard deviation of the gaussian, by default 0.25

        """
        super().__init__()

        self._loc = mean
        self._scale = std

    def sample(self, x: float) -> float:
        """Sample the prior.

        Parameters
        ----------
        x : float
            A random number between 0 and 1.

        """
        return stats.norm.ppf(x, loc=self._loc, scale=self._scale)

    def params(self) -> str:
        """Return the parameters of the prior in string form."""
        return f"Mean = {self._loc} Stdev = {self._scale}"

    def boundaries(self) -> t.Tuple[float, float]:
        """Return the boundaries of the prior."""
        return self.sample(0.1), self.sample(0.9)


class LogGaussian(Gaussian):
    """Defines a log-gaussian prior."""

    def __init__(
        self,
        mean: t.Optional[float] = 0.5,
        std: t.Optional[float] = 0.25,
        lin_mean: t.Optional[float] = None,
        lin_std: t.Optional[float] = None,
    ):
        """Initialise log-gaussian prior.

        Parameters
        ----------
        mean : float, optional
            Mean of the gaussian, by default 0.5
        std : float, optional
            Standard deviation of the gaussian, by default 0.25
        lin_mean : float, optional
            Mean of the gaussian in linear space, by default None
        lin_std : float, optional
            Standard deviation of the gaussian in linear space, by default None

        """
        if lin_mean is not None:
            mean = math.log10(lin_mean)
        if lin_std is not None:
            std = math.log10(lin_std)
        super().__init__(mean=mean, std=std)
        self._prior_mode = PriorMode.LOG

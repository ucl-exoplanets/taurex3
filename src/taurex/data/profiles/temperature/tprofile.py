"""Base temperature class."""
import typing as t

import numpy as np
import numpy.typing as npt

from taurex.data.citation import Citable
from taurex.data.fittable import Fittable, derivedparam
from taurex.log import Logger
from taurex.output import OutputGroup
from taurex.output.writeable import Writeable
from taurex.planet import Planet


class TemperatureProfile(Fittable, Logger, Writeable, Citable):
    """Defines temperature profile for an atmosphere.

    *Abstract Class*

    Must define:

    - :func:`profile`

    """

    def __init__(self, name: str):
        """Initialize temperature class.

        Parameters
        ----------
        name : str
            Name used in logging

        """
        Logger.__init__(self, name)
        Fittable.__init__(self)

    def initialize_profile(
        self,
        planet: t.Optional[Planet] = None,
        nlayers: t.Optional[int] = 100,
        pressure_profile: t.Optional[npt.NDArray] = None,
    ):
        """Initializes the profile.

        Parameters
        ----------
        planet: :class:`~taurex.data.planet.Planet`

        nlayers: int
            Number of layers in atmosphere

        pressure_profile: :obj:`array`
            Pressure at each layer of the atmosphere

        """
        self.nlayers = nlayers
        self.nlevels = nlayers + 1
        self.pressure_profile = pressure_profile
        self.planet = planet

    @property
    def profile(self) -> npt.NDArray[np.float64]:
        """Temperature profile at each layer.

        Must return a temperature profile at each layer of the atmosphere

        Returns
        -------
        temperature: :obj:`array`
            Temperature in Kelvin
        """
        raise NotImplementedError

    def write(self, output: OutputGroup) -> OutputGroup:
        """Write temperature profile to output."""
        temperature = output.create_group("Temperature")
        temperature.write_string("temperature_type", self.__class__.__name__)

        return temperature

    @derivedparam(param_name="avg_T", param_latex="$\\bar{T}$", compute=False)
    def averageTemperature(self) -> float:  # noqa: N802
        """Average temperature across all layers."""
        return np.mean(self.profile)

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Return all input keywords."""
        raise NotImplementedError

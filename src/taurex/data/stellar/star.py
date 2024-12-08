"""Base stellar class."""
import typing as t

import numpy as np
import numpy.typing as npt

from taurex.constants import MSOL, RSOL
from taurex.data.fittable import Fittable
from taurex.log import Logger
from taurex.output import OutputGroup, Writeable
from taurex.util.emission import black_body

from ..citation import Citable
from ..fittable import Fittable, derivedparam, fitparam


class Star(Fittable, Logger, Writeable, Citable):
    """A base class that holds information on the star in the model.
    Its implementation is a star that has a blackbody spectrum.

    """

    def __init__(
        self,
        temperature: t.Optional[float] = 5000,
        radius: t.Optional[float] = 1.0,
        distance: t.Optional[float] = 1,
        magnitudeK: t.Optional[float] = 10.0,  # noqa: N803
        mass: t.Optional[float] = 1.0,
        metallicity: t.Optional[float] = 1.0,
    ):
        """Initialize a star.

        Parameters
        ----------

        temperature: float, optional
            Stellar temperature in Kelvin

        radius: float, optional
            Stellar radius in Solar radius

        metallicity: float, optional
            Metallicity in solar values

        mass: float, optional
            Stellar mass in solar mass

        distance: float, optional
            Distance from Earth in pc

        magnitudeK: float, optional
            Maginitude in K band

        """
        Logger.__init__(self, self.__class__.__name__)
        Fittable.__init__(self)
        self._temperature = temperature
        self._radius = radius * RSOL
        self._mass = mass * MSOL
        self.debug("Star mass %s", self._mass)
        self.sed = None
        self.distance = distance
        self.magnitudeK = magnitudeK
        self._metallicity = metallicity

    @property
    def radius(self) -> float:
        """Radius in metres."""
        return self._radius

    @property
    def temperature(self) -> float:
        """Blackbody temperature in Kelvin."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        """Set blackbody temperature in Kelvin."""
        self._temperature = value

    @property
    def mass(self) -> float:
        """Mass in kg."""
        return self._mass

    @fitparam(
        param_name="distance",
        param_latex="$distance$",
        default_fit=False,
        default_bounds=[1, 22],
    )
    def distanceSystem(self) -> float:
        """Distance from Earth to the System (in pc)."""
        return self.distance

    @distanceSystem.setter
    def distanceSystem(self, value: float) -> None:
        """Set distance from Earth to System (in pc)."""
        self.distance = value

    def initialize(self, wngrid: npt.NDArray[np.float64]) -> None:
        """Initializes the blackbody spectrum on the given wavenumber grid

        Parameters
        ----------
        wngrid: :obj:`array`
            Wavenumber grid cm-1 to compute black body spectrum

        """
        self.sed = black_body(wngrid, self.temperature)

    @property
    def spectralEmissionDensity(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Spectral emmision density in W/m2/cm-1/sr."""
        return self.sed

    def write(self, output: OutputGroup) -> OutputGroup:
        """Write to output group."""
        star = output.create_group("Star")
        star.write_string("star_type", self.__class__.__name__)
        star.write_scalar("temperature", self.temperature)
        star.write_scalar("radius", self._radius / RSOL)
        star.write_scalar("distance", self.distance)
        star.write_scalar("mass", self._mass / MSOL)
        star.write_scalar("magnitudeK", self.magnitudeK)
        star.write_scalar("metallicity", self._metallicity)
        star.write_scalar("radius_m", self.radius)
        star.write_array("SED", self.spectralEmissionDensity)
        star.write_scalar("mass_kg", self._mass)
        return star

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Input keywords for star."""
        raise NotImplementedError


class BlackbodyStar(Star):
    """Alias for the base star type"""

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        return ("blackbody",)

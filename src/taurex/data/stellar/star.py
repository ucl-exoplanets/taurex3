"""Base stellar class."""

import typing as t

import numpy as np
import numpy.typing as npt
from astropy import units as u

from taurex.constants import MSOL
from taurex.constants import RSOL
from taurex.data.fittable import Fittable
from taurex.data.fittable import fitparam
from taurex.log import Logger
from taurex.output import OutputGroup
from taurex.output import Writeable
from taurex.util import convert_to_unit_value
from taurex.util.emission import black_body

from ..citation import Citable


class Star(Fittable, Logger, Writeable, Citable):
    """A base class that holds information on the star in the model.

    Its implementation is a star that has a blackbody spectrum.

    """

    def __init__(
        self,
        temperature: t.Optional[t.Union[float, u.Quantity]] = 5000,
        radius: t.Optional[t.Union[float, u.Quantity]] = 1.0,
        distance: t.Optional[t.Union[float, u.Quantity]] = 1,
        magnitudeK: t.Optional[float] = 10.0,  # noqa: N803
        mass: t.Optional[t.Union[float, u.Quantity]] = 1.0,
        metallicity: t.Optional[t.Union[float, u.Quantity]] = 1.0,
    ):
        """Initialize a star.

        Parameters
        ----------
        temperature: float or astropy.units.Quantity, optional
            Stellar temperature in K when unitless.

        radius: float or astropy.units.Quantity, optional
            Stellar radius in Solar radii when unitless.

        metallicity: float or astropy.units.Quantity, optional
            Metallicity in solar values when unitless.

        mass: float or astropy.units.Quantity, optional
            Stellar mass in Solar masses when unitless.

        distance: float or astropy.units.Quantity, optional
            Distance from Earth in pc when unitless.

        magnitudeK: float, optional
            Maginitude in K band

        """
        Logger.__init__(self, self.__class__.__name__)
        Fittable.__init__(self)
        self._temperature = self._normalize_temperature(temperature)
        self._radius = convert_to_unit_value(radius, u.m, default_unit=u.R_sun)
        self._mass = convert_to_unit_value(mass, u.kg, default_unit=u.M_sun)
        self.debug("Star mass %s", self._mass)
        self.sed = None
        self.distance = self._normalize_distance(distance)
        self.magnitudeK = magnitudeK
        self._metallicity = self._normalize_dimensionless(metallicity)

    @staticmethod
    def _normalize_temperature(value: t.Union[float, u.Quantity]) -> float:
        """Convert temperature to plain numeric values in K."""
        return convert_to_unit_value(
            value, u.K, default_unit=u.K, equivalencies=u.temperature()
        )

    @staticmethod
    def _normalize_distance(value: t.Union[float, u.Quantity]) -> float:
        """Convert distance to plain numeric values in pc."""
        return convert_to_unit_value(value, u.pc, default_unit=u.pc)

    @staticmethod
    def _normalize_dimensionless(value: t.Union[float, u.Quantity]) -> float:
        """Convert a dimensionless stellar input to a plain numeric value."""
        return convert_to_unit_value(
            value,
            u.dimensionless_unscaled,
            default_unit=u.dimensionless_unscaled,
        )

    @property
    def radius(self) -> float:
        """Radius in metres."""
        return self._radius

    @property
    def temperature(self) -> float:
        """Blackbody temperature in Kelvin."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: t.Union[float, u.Quantity]) -> None:
        """Set blackbody temperature in Kelvin."""
        self._temperature = self._normalize_temperature(value)

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
    def distanceSystem(self) -> float:  # noqa: N802
        """Distance from Earth to the System (in pc)."""
        return self.distance

    @distanceSystem.setter
    def distanceSystem(self, value: t.Union[float, u.Quantity]) -> None:  # noqa: N802
        """Set distance from Earth to System (in pc)."""
        self.distance = self._normalize_distance(value)

    def initialize(self, wngrid: npt.NDArray[np.float64]) -> None:
        """Initializes the blackbody spectrum on the given wavenumber grid.

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
        """Write to output group.

        Parameters
        ----------
        output : :class:`~taurex.output.output.OutputGroup`
            Output group to write to.

        Returns
        -------
        :class:`~taurex.output.output.OutputGroup`

        """
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
    """Alias for the base star type."""

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Input keywords for Blackbody star."""
        return ("blackbody",)

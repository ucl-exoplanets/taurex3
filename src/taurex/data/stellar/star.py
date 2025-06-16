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

    @radius.setter
    def radius(self, value: float) -> None:
        """Set radius in metres."""
        self._radius = value

    

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

    @mass.setter
    def mass(self, value: float) -> None:
        """Set mass in kg."""
        self._mass = value


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

    @fitparam(
        param_name="star_radius",
        param_latex="$R_{\\star}$",
        default_fit=False,
        default_bounds=[0.1, 10],
    )
    def radiusRsol(self) -> float:
        """Radius in Solar radius."""
        return self._radius / RSOL
    
    @radiusRsol.setter
    def radiusRsol(self, value: float) -> None:
        """Set radius in Solar radius."""
        self._radius = value * RSOL

    @fitparam(
        param_name="star_mass",
        param_latex="$M_{\\star}$",
        default_fit=False,
        default_bounds=[0.1, 10],
    )
    def massMsol(self) -> float:
        """Mass in Solar mass."""
        return self._mass / MSOL
    
    @massMsol.setter
    def massMsol(self, value: float) -> None:
        """Set mass in Solar mass."""
        self._mass = value * MSOL

    @fitparam(
        param_name="star_metallicity",
        param_latex="$Z$",
        default_fit=False,
        default_mode="log",
        default_bounds=[0.01, 10],
    )
    def metallicityZ(self) -> float:
        """Metallicity in Solar values."""
        return self._metallicity
    
    @metallicityZ.setter
    def metallicityZ(self, value: float) -> None:
        """Set metallicity in Solar values."""
        self._metallicity = value

    @fitparam(
        param_name="star_temperature",
        param_latex="$T_{\\star}$",
        default_fit=False,
        default_bounds=[1000, 20000],
    )
    def temperatureK(self) -> float:
        """Temperature in Kelvin."""
        return self._temperature
    
    @temperatureK.setter
    def temperatureK(self, value: float) -> None:
        """Set temperature in Kelvin."""
        self._temperature = value
    
    @derivedparam(
        param_name="star_logg",
        param_latex="$\\log_{10} g_{\\star}$",
        compute=False,
    )
    def logg(self) -> float:
        """Logarithm of surface gravity in cgs."""
        # logg = G * M / R^2
        # where G is the gravitational constant, M is the mass, and R is the radius
        from astropy import constants as const
        from astropy import units as u
        import math

        radius = self._radius << u.m
        mass = self._mass << u.kg

        grav = (const.G * mass / (radius ** 2)).to(u.cm/u.s**2).value
        self.debug("Computed surface gravity: %s", grav)
        # Convert to log10

        return math.log10(grav)


        G = 6.67430e-11
        logg_value = np.log10(G * self._mass / (self._radius ** 2))
        self.debug("Computed logg: %s", logg_value)
        return logg_value


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

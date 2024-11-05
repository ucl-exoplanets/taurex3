"""Module for all things planet."""
import math
import typing as t
from warnings import warn

import numpy as np
import numpy.typing as npt

from taurex.constants import AU, MJUP, RJUP, G
from taurex.log import Logger
from taurex.output import OutputGroup
from taurex.output.writeable import Writeable
from taurex.util import conversion_factor

from .citation import Citable
from .fittable import Fittable, derivedparam, fitparam


class BasePlanet(Fittable, Logger, Writeable, Citable):
    """Base Planet class.

    Holds information on a planet and its properties and
    derived properties

    """

    def __init__(
        self,
        planet_mass: t.Optional[float] = 1.0,
        planet_radius: t.Optional[float] = 1.0,
        planet_sma: t.Optional[float] = None,
        planet_distance: t.Optional[float] = 1.0,
        impact_param: t.Optional[float] = 0.5,
        orbital_period: t.Optional[float] = 2.0,
        albedo: t.Optional[float] = 0.3,
        transit_time: t.Optional[float] = 3000.0,
    ) -> None:
        """Initialise a planet.

        Parameters
        -----------

        planet_mass: float, optional
            mass in terms of Jupiter mass of the planet
        planet_radius: float, optional
            radius in terms of Jupiter radii of the planet
        planet_sma: float, optional
            Semi-major axis in AU
        impact_param: float, optional
            Impact parameter
        orbital_period: float, optional
            Orbital period in days
        albedo: float, optional
            Planetary albedo
        transit_time: float, optional
            Transit time in seconds

        """
        Logger.__init__(self, "Planet")
        Fittable.__init__(self)
        self.set_planet_mass(planet_mass, "Mjup")
        self.set_planet_radius(planet_radius, "Rjup")
        self.set_planet_semimajoraxis(planet_sma or planet_distance)
        self._impact = impact_param
        self._orbit_period = orbital_period
        self._albedo = albedo
        self._transit_time = transit_time

    def set_planet_radius(self, value: float, unit: t.Optional[str] = "Rjup") -> None:
        """Set planet radius."""
        factor = conversion_factor(unit, "m")
        self._radius = value * factor

    def set_planet_mass(self, value: float, unit="Mjup") -> None:
        """Set planet mass."""
        factor = conversion_factor(unit, "kg")
        self._mass = value * factor

    def set_planet_semimajoraxis(self, value: float, unit="AU") -> None:
        """Set planet semi major axis."""
        factor = conversion_factor(unit, "m")
        self._distance = value * factor

    def get_planet_radius(self, unit: t.Optional[str] = "Rjup") -> float:
        """Get planet radius in specified unit (default is Rjup)."""
        factor = conversion_factor("m", unit)
        return self._radius * factor

    def get_planet_mass(self, unit: t.Optional[str] = "Mjup") -> float:
        """Get planet mass in specified unit (default is Mjup)."""
        factor = conversion_factor("kg", unit)
        return self._mass * factor

    def get_planet_semimajoraxis(self, unit: t.Optional[str] = "AU") -> float:
        """Get planet semi major axis in specified unit (default is AU)."""
        factor = conversion_factor("m", unit)
        return self._distance * factor

    @fitparam(
        param_name="planet_mass",
        param_latex="$M_p$",
        default_fit=False,
        default_bounds=[0.5, 1.5],
    )
    def mass(self) -> float:
        """Planet mass in Jupiter mass."""
        return self.get_planet_mass(unit="Mjup")

    @mass.setter
    def mass(self, value: float) -> None:
        """Set planet mass in Jupiter mass."""
        self.set_planet_mass(value, unit="Mjup")

    @fitparam(
        param_name="planet_radius",
        param_latex="$R_p$",
        default_fit=False,
        default_bounds=[0.9, 1.1],
    )
    def radius(self) -> float:
        """Planet radius in Jupiter radii."""
        return self.get_planet_radius(unit="Rjup")

    @radius.setter
    def radius(self, value: float) -> None:
        """Set planet radius in Jupiter radii."""
        self.set_planet_radius(value, unit="Rjup")

    @property
    def fullRadius(self) -> float:  # noqa: N802
        """Planet radius in metres

        Deprecated, use :func:`get_planet_radius` instead

        """
        warn(
            "fullRadius is deprecated, use get_planet_radius(unit='m') instead",
            DeprecationWarning,
        )
        return self.get_planet_radius(unit="m")

    @property
    def fullMass(self) -> float:  # noqa: N802
        """Planet mass in kg.

        Deprecated, use :func:`get_planet_mass` instead

        """
        warn(
            "fullMass is deprecated, use get_planet_mass(unit='kg') instead",
            DeprecationWarning,
        )
        return self._mass

    @property
    def impactParameter(self) -> float:  # noqa: N802
        """Planet impact parameter."""
        return self._impact

    @property
    def orbitalPeriod(self) -> float:  # noqa: N802
        """Planet orbital period in days."""
        return self._orbit_period

    @property
    def albedo(self) -> float:  # noqa: N802
        """Planet albedo."""
        return self._albedo

    @property
    def transitTime(self) -> float:  # noqa: N802
        """Planet transit time in seconds."""
        return self._transit_time

    @fitparam(
        param_name="planet_distance",
        param_latex="$D_{planet}$",
        default_fit=False,
        default_bounds=[1, 2],
    )
    def distance(self) -> float:
        """Planet semi major axis from parent star (AU)."""
        return self.get_planet_semimajoraxis(unit="AU")

    @distance.setter
    def distance(self, value: float) -> None:
        """Set planet semi major axis from parent star (AU)."""
        self.set_planet_semimajoraxis(value, unit="AU")

    @fitparam(
        param_name="planet_sma",
        param_latex="$D_{planet}$",
        default_fit=False,
        default_bounds=[1, 2],
    )
    def semiMajorAxis(self) -> float:  # noqa: N802
        """Planet semi major axis from parent star (AU) (ALIAS)"""
        return self.get_planet_semimajoraxis(unit="AU")

    @semiMajorAxis.setter
    def semiMajorAxis(self, value: float) -> None:  # noqa: N802
        """Set planet semi major axis from parent star (AU) (ALIAS)"""
        self.set_planet_semimajoraxis(value, unit="AU")

    @property
    def gravity(self) -> float:
        """Surface gravity in ms-2."""
        return (G * self.get_planet_mass(unit="kg")) / (
            self.get_planet_radius(unit="m") ** 2
        )

    def gravity_at_height(self, height: float) -> float:
        """Gravity at height (m) from planet in ms-2

        Parameters
        ----------
        height: float
            Height in metres from planet surface

        Returns
        -------
        g: float
            Gravity in ms-2

        """
        return (G * self.get_planet_mass(unit="kg")) / (
            (self.get_planet_radius(unit="m") + height) ** 2
        )

    def write(self, output: OutputGroup) -> OutputGroup:
        """Write planet information to output group."""
        planet = output.create_group("Planet")

        planet.write_string("planet_type", self.__class__.__name__)
        planet.write_scalar("planet_mass", self._mass / MJUP)
        planet.write_scalar("planet_radius", self._radius / RJUP)
        planet.write_scalar("planet_distance", self._distance / AU)
        planet.write_scalar("impact_param", self._impact)
        planet.write_scalar("orbital_period", self.orbitalPeriod)
        planet.write_scalar("albedo", self.albedo)
        planet.write_scalar("transit_time", self.transitTime)

        planet.write_scalar("mass_kg", self.mass)
        planet.write_scalar("radius_m", self.radius)
        planet.write_scalar("surface_gravity", self.gravity)
        return planet

    @derivedparam(param_name="logg", param_latex="log(g)", compute=False)
    def logg(self) -> float:
        """Surface gravity (m2/s) in log10"""
        return math.log10(self.gravity)

    def calculate_scale_properties(
        self,
        temperature: npt.NDArray[np.float64],
        pressure_levels: npt.NDArray[np.float64],
        mu: npt.NDArray[np.float64],
        length_units: t.Optional[str] = "m",
    ) -> t.Tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Solve hydrostatic for altitude, gravity and scale height of the atmosphere.

        Parameters
        ----------
        temperature: array_like
            Ttaurex.utilch layer in K

        pressure_levels: array_like
            Pressure at each layer boundary in Pa

        mu: array_like
            mean moleculer weight for each layer in kg

        Returns
        -------
        z: array
            Altitude at each layer boundary
        H: array
            scale height converted to correct length units
        g: array
            gravity converted to correct length units
        deltaz:
            dz in length units

        """

        from taurex.constants import KBOLTZ
        from taurex.util import conversion_factor

        # build the altitude profile from the bottom up
        nlayers = temperature.shape[0]
        scaleheight = np.zeros(nlayers)
        g = np.zeros(nlayers)
        z = np.zeros(nlayers + 1)
        deltaz = np.zeros(nlayers + 1)

        # surface gravity (0th layer)
        g[0] = self.gravity
        # scaleheight at the surface (0th layer)
        scaleheight[0] = (KBOLTZ * temperature[0]) / (mu[0] * g[0])
        #####
        ####
        ####

        factor = conversion_factor("m", length_units)

        for i in range(1, nlayers + 1):
            deltaz[i] = (
                (-1.0)
                * scaleheight[i - 1]
                * np.log(pressure_levels[i] / pressure_levels[i - 1])
            )
            z[i] = z[i - 1] + deltaz[i]
            if i < nlayers:
                with np.errstate(over="ignore"):
                    # gravity at the i-th layer
                    g[i] = self.gravity_at_height(z[i])
                    self.debug("G[%s] = %s", i, g[i])

                with np.errstate(divide="ignore"):
                    scaleheight[i] = (KBOLTZ * temperature[i]) / (mu[i] * g[i])

        return z * factor, scaleheight * factor, g * factor, deltaz[1:] * factor

    def compute_path_length(
        self,
        altitudes: npt.NDArray[np.float64],
        viewer: npt.NDArray[np.float64],
        tangent: npt.NDArray[np.float64],
        vector_coord_sys: t.Optional[t.Literal["cartesian"]] = "cartesian",
    ):
        """Compute path length through atmosphere."""
        from taurex.util.geometry import compute_path_length_3d

        result = compute_path_length_3d(
            self.fullRadius, altitudes, viewer, tangent, coordinates=vector_coord_sys
        )

        return result

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Return list of keywords for input file."""
        raise NotImplementedError


class Planet(BasePlanet):
    """Planet class."""

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Return list of keywords for input file."""
        return (
            "simple",
            "sphere",
        )


class Earth(Planet):
    """An implementation for earth"""

    def __init__(self):
        """Initialise earth"""
        Planet.__init__(self, 5.972e24, 6371000)

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Return list of keywords for input file."""
        return ("earth",)


class Mars(Planet):
    """An implementation for mars."""

    def __init__(self) -> None:
        """Initialise mars."""
        import astropy.units as u

        radius = (0.532 * u.R_earth).to(u.jupiterRad)
        mass = (0.107 * u.M_earth).to(u.jupiterMass)
        distance = 1.524
        Planet.__init__(mass=mass.value, radius=radius.value, distance=distance)

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        return ("mars",)

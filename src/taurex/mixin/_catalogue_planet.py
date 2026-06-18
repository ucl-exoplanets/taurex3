"""Planet mixins that load parameters from a catalogue (file or ExoMAST)."""

import requests

from taurex.constants import MJUP
from taurex.constants import RJUP
from taurex.mixin import PlanetMixin

from ._catalogue_reader import FileReader


# ---------------------------------------------------------------------------
#  File-based planet mixin
# ---------------------------------------------------------------------------


class PlanetCatalogueFile(PlanetMixin):
    """Planet mixin that populates parameters from a local CSV/TSV file.

    >>> from taurex.mixin import enhance_class
    >>> from taurex.planet import Planet
    >>> from taurex.mixin import PlanetCatalogueFile
    >>> PlanetF = enhance_class(Planet, PlanetCatalogueFile,
    ...                         catalogue_file="targets.csv")
    """

    def __init_mixin__(
        self,
        planet_name=None,
        planet_mass=None,
        planet_radius=None,
        planet_distance=None,
        impact_param=0.5,
        orbital_period=None,
        albedo=0.3,
        transit_time=None,
        eccentricity=None,
        pericentre_long=None,
        pericentre_time=None,
        ascending_node_long=None,
        mid_time=None,
        inclination=None,
        catalogue_file=None,
        planet_no=0,
    ):
        """Populate planet parameters from a CSV file via the mixin."""
        self._catalogue_file = catalogue_file
        self._planet_no = planet_no

        if catalogue_file is not None:
            reader = FileReader(
                filename=catalogue_file,
                target_no=planet_no,
                target_name=planet_name,
            )
            for field, value, _unit_str in reader.planet_params:
                field_lower = field.lower()
                if "mass" in field_lower and planet_mass is None:
                    planet_mass = float(value)
                elif "radius" in field_lower and planet_radius is None:
                    planet_radius = float(value)
                elif "semi-major" in field_lower and planet_distance is None:
                    planet_distance = float(value)
                elif "period" in field_lower and orbital_period is None:
                    orbital_period = float(value)
                elif "inclination" in field_lower and inclination is None:
                    inclination = float(value)
                elif "eccentricity" in field_lower and eccentricity is None:
                    eccentricity = float(value)
                elif "transit" in field_lower and transit_time is None:
                    transit_time = float(value)

        # Set defaults from base
        self.set_planet_mass(planet_mass or 1.0, "Mjup")
        self.set_planet_radius(planet_radius or 1.0, "Rjup")
        self.set_planet_semimajoraxis(planet_distance or 1.0)
        self._impact = impact_param
        self._albedo = albedo
        self._orbit_period = orbital_period or 2.0
        self._transit_time = transit_time or 3000.0
        self._eccentricity = eccentricity or 0.0
        self._pericentre_long = pericentre_long or 0.0
        self._pericentre_time = pericentre_time or 0.0
        self._ascending_node_long = ascending_node_long or 0.0
        self._mid_time = mid_time or 0.0
        self._inclination = inclination or 90.0

    @classmethod
    def input_keywords(cls):
        """Return the class factory keyword."""
        return ("planetcataloguefile",)


# ---------------------------------------------------------------------------
#  ExoMAST planet mixin
# ---------------------------------------------------------------------------


class PlanetCatalogueExomast(PlanetMixin):
    """Planet mixin that populates parameters from the ExoMAST API.

    >>> from taurex.mixin import enhance_class
    >>> from taurex.planet import Planet
    >>> from taurex.mixin import PlanetCatalogueExomast
    >>> PlanetE = enhance_class(Planet, PlanetCatalogueExomast,
    ...                         planet_name="WASP-121 b")
    """

    def __init_mixin__(  # noqa: C901
        self,
        planet_name=None,
        planet_mass=None,
        planet_radius=None,
        planet_distance=None,
        impact_param=0.5,
        orbital_period=None,
        albedo=0.3,
        transit_time=None,
        eccentricity=None,
        pericentre_long=None,
        pericentre_time=None,
        ascending_node_long=None,
        mid_time=None,
        inclination=None,
        planet_no=0,
    ):
        """Populate planet parameters from the ExoMAST API."""
        if planet_name is not None and planet_mass is None:
            pmass, prad, pdist, pper, pincl, pecc, ptrans = (
                planet_mass, planet_radius, planet_distance,
                orbital_period, inclination, eccentricity, transit_time,
            )
            try:
                data = self._fetch_exomast(planet_name)
                if pmass is None and "pl_masse" in data:
                    val = float(data["pl_masse"][0])
                    pmass = (
                        val * MJUP.value
                        if data.get("pl_masse_unit") == "Mj"
                        else val
                    )
                if prad is None and "pl_rade" in data:
                    val = float(data["pl_rade"][0])
                    prad = (
                        val * RJUP.value
                        if data.get("pl_rade_unit") == "Rj"
                        else val
                    )
                if pdist is None and "pl_orbsmax" in data:
                    pdist = float(data["pl_orbsmax"][0])
                if pper is None and "pl_orbper" in data:
                    pper = float(data["pl_orbper"][0])
                if pincl is None and "pl_orbincl" in data:
                    pincl = float(data["pl_orbincl"][0])
                if pecc is None and "pl_orbeccen" in data:
                    pecc = float(data["pl_orbeccen"][0])
                if ptrans is None and "pl_trandur" in data:
                    ptrans = float(data["pl_trandur"][0])
            except requests.RequestException:
                pass
            planet_mass, planet_radius, planet_distance = pmass, prad, pdist
            orbital_period, inclination, eccentricity = pper, pincl, pecc
            transit_time = ptrans

        self.set_planet_mass(planet_mass or 1.0, "Mjup")
        self.set_planet_radius(planet_radius or 1.0, "Rjup")
        self.set_planet_semimajoraxis(planet_distance or 1.0)
        self._impact = impact_param
        self._albedo = albedo
        self._orbit_period = orbital_period or 2.0
        self._transit_time = transit_time or 3000.0
        self._eccentricity = eccentricity or 0.0
        self._pericentre_long = pericentre_long or 0.0
        self._pericentre_time = pericentre_time or 0.0
        self._ascending_node_long = ascending_node_long or 0.0
        self._mid_time = mid_time or 0.0
        self._inclination = inclination or 90.0

    @staticmethod
    def _fetch_exomast(planet_name):
        """Fetch planetary data from the ExoMAST API."""
        url = "https://exo.mast.stsci.edu/api/v1.0/exoplanets/"
        r = requests.get(f"{url}{planet_name}", timeout=30)
        r.raise_for_status()
        return r.json()

    @classmethod
    def input_keywords(cls):
        """Return the class factory keyword."""
        return ("planetcatalogueexomast",)

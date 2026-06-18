"""Star mixins that load parameters from a catalogue (file or ExoMAST)."""

import requests

from taurex.constants import MSOL
from taurex.constants import RSOL
from taurex.mixin import StarMixin

from ._catalogue_reader import FileReader


# ---------------------------------------------------------------------------
#  File-based star mixin
# ---------------------------------------------------------------------------


class StarCatalogueFile(StarMixin):
    """Star mixin that populates parameters from a local CSV/TSV file.

    >>> from taurex.mixin import enhance_class
    >>> from taurex.stellar import BlackbodyStar
    >>> from taurex.mixin import StarCatalogueFile
    >>> StarF = enhance_class(BlackbodyStar, StarCatalogueFile,
    ...                       catalogue_file="targets.csv")
    """

    def __init_mixin__(
        self,
        planet_name=None,
        temperature=None,
        radius=None,
        distance=None,
        magnitudeK=10.0,
        metallicity=None,
        mass=None,
        ld_method="unique",
        ldc=None,
        catalogue_file=None,
        planet_no=0,
    ):
        """Populate star parameters from a CSV file."""
        if catalogue_file is not None:
            reader = FileReader(
                filename=catalogue_file,
                target_no=planet_no,
                target_name=planet_name,
            )
            for field, value, _unit_str in reader.star_params:
                field_lower = field.lower()
                if "temperature" in field_lower and temperature is None:
                    temperature = float(value)
                elif "radius" in field_lower and radius is None:
                    radius = float(value)
                elif "distance" in field_lower and distance is None:
                    distance = float(value)
                elif "metallicity" in field_lower and metallicity is None:
                    metallicity = float(value)
                elif "mass" in field_lower and mass is None:
                    mass = float(value)

        self._temperature = temperature or 5000
        self._radius = (radius or 1.0) * RSOL
        self._mass = (mass or 1.0) * MSOL
        self.distance = distance or 1.0
        self.magnitudeK = magnitudeK
        self._metallicity = metallicity or 0.0

    @classmethod
    def input_keywords(cls):
        """Return the class factory keyword."""
        return ("starcataloguefile",)


# ---------------------------------------------------------------------------
#  ExoMAST star mixin
# ---------------------------------------------------------------------------


class StarCatalogueExomast(StarMixin):
    """Star mixin that populates parameters from the ExoMAST API.

    >>> from taurex.mixin import enhance_class
    >>> from taurex.stellar import BlackbodyStar
    >>> from taurex.mixin import StarCatalogueExomast
    >>> StarE = enhance_class(BlackbodyStar, StarCatalogueExomast,
    ...                       planet_name="WASP-121 b")
    """

    def __init_mixin__(
        self,
        planet_name=None,
        temperature=None,
        radius=None,
        distance=None,
        magnitudeK=10.0,
        metallicity=None,
        mass=None,
        ld_method="unique",
        ldc=None,
        planet_no=0,
    ):
        """Populate star parameters from the ExoMAST API."""
        if planet_name is not None and temperature is None:
            data = self._fetch_exomast(planet_name)
            if temperature is None and "st_teff" in data:
                temperature = float(data["st_teff"][0])
            if radius is None and "st_rad" in data:
                radius = float(data["st_rad"][0])
            if mass is None and "st_mass" in data:
                mass = float(data["st_mass"][0])
            if metallicity is None and "st_met" in data:
                metallicity = float(data["st_met"][0])
            if distance is None and "sy_dist" in data:
                distance = float(data["sy_dist"][0])

        self._temperature = temperature or 5000
        self._radius = (radius or 1.0) * RSOL
        self._mass = (mass or 1.0) * MSOL
        self.distance = distance or 1.0
        self.magnitudeK = magnitudeK
        self._metallicity = metallicity or 0.0

    @staticmethod
    def _fetch_exomast(planet_name):
        """Fetch stellar data from the ExoMAST API."""
        url = "https://exo.mast.stsci.edu/api/v1.0/exoplanets/"
        r = requests.get(f"{url}{planet_name}", timeout=30)
        r.raise_for_status()
        return r.json()

    @classmethod
    def input_keywords(cls):
        """Return the class factory keyword."""
        return ("starcatalogueexomast",)

"""Star mixins that load parameters from a catalogue (file or ExoMAST)."""

from taurex.constants import MSOL
from taurex.constants import RSOL

from ._catalogue_reader import FileReader
from ._catalogue_reader import fetch_exomast_properties
from .core import StarMixin


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
            if temperature is None and "Teff" in data:
                temperature = float(data["Teff"])
            if radius is None and "Rs" in data:
                radius = float(data["Rs"])
            if mass is None and "Ms" in data:
                mass = float(data["Ms"])
            if metallicity is None and "Fe/H" in data:
                metallicity = float(data["Fe/H"])
            if distance is None and "distance" in data:
                distance = float(data["distance"])

        self._temperature = temperature or 5000
        self._radius = (radius or 1.0) * RSOL
        self._mass = (mass or 1.0) * MSOL
        self.distance = distance or 1.0
        self.magnitudeK = magnitudeK
        self._metallicity = metallicity or 0.0

    @staticmethod
    def _fetch_exomast(planet_name):
        """Fetch stellar data from the ExoMAST API."""
        return fetch_exomast_properties(planet_name)

    @classmethod
    def input_keywords(cls):
        """Return the class factory keyword."""
        return ("starcatalogueexomast",)

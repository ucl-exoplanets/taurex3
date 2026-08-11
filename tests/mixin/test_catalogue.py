"""Tests for the catalogue mixins (file-based and ExoMAST)."""

import csv
from unittest.mock import patch

import pytest

from taurex.mixin import enhance_class
from taurex.mixin._catalogue_planet import PlanetCatalogueExomast
from taurex.mixin._catalogue_planet import PlanetCatalogueFile
from taurex.mixin._catalogue_reader import FileReader
from taurex.mixin._catalogue_star import StarCatalogueExomast
from taurex.mixin._catalogue_star import StarCatalogueFile
from taurex.planet import Planet
from taurex.stellar import BlackbodyStar


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _write_csv(path, headers, rows):
    """Write a small CSV file with the given headers and a single data row."""
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
#  FileReader tests
# ---------------------------------------------------------------------------


class TestFileReader:
    """Tests for the FileReader catalogue parser."""

    def test_basic_read(self, tmp_path):
        """Read a simple CSV with star and planet columns."""
        csv_path = tmp_path / "catalogue.csv"
        _write_csv(
            csv_path,
            [
                "star_temperature [K]",
                "star_radius [R_sun]",
                "star_mass [M_sun]",
                "planet_radius [R_jup]",
                "planet_mass [M_jup]",
            ],
            [[5800, 1.0, 1.0, 1.2, 0.8]],
        )

        reader = FileReader(filename=str(csv_path), target_no=0)
        assert len(reader.star_params) == 3
        assert len(reader.planet_params) == 2

        # Check star params: (field, value, unit)
        star_fields = {p[0]: p for p in reader.star_params}
        assert star_fields["star_temperature [K]"][1] == 5800
        assert star_fields["star_temperature [K]"][2] == "K"
        assert star_fields["star_radius [R_sun]"][2] == "R_sun"

        # Check planet params
        planet_fields = {p[0]: p for p in reader.planet_params}
        assert planet_fields["planet_radius [R_jup]"][1] == 1.2
        assert planet_fields["planet_mass [M_jup]"][1] == 0.8

    def test_target_by_name(self, tmp_path):
        """Look up a target by name instead of row index (single row)."""
        csv_path = tmp_path / "catalogue.csv"
        _write_csv(
            csv_path,
            ["planet_name", "star_temperature [K]", "planet_radius [R_jup]"],
            [["HD 209458 b", 6000, 1.4]],
        )

        reader = FileReader(
            filename=str(csv_path), target_no=0, target_name="HD 209458"
        )
        star = {p[0]: p for p in reader.star_params}
        assert star["star_temperature [K]"][1] == 6000


# ---------------------------------------------------------------------------
#  PlanetCatalogueFile tests
# ---------------------------------------------------------------------------


class TestPlanetCatalogueFile:
    """Tests for the file-based planet catalogue mixin."""

    def test_populates_from_csv(self, tmp_path):
        """Planet parameters are loaded from a CSV catalogue file."""
        csv_path = tmp_path / "planets.csv"
        _write_csv(
            csv_path,
            [
                "planet_mass [M_jup]",
                "planet_radius [R_jup]",
                "planet_semi-major_axis [AU]",
                "planet_period [days]",
                "planet_inclination [deg]",
                "planet_eccentricity",
                "planet_transit_duration [s]",
            ],
            [[1.5, 1.2, 0.05, 3.5, 87.0, 0.01, 7200]],
        )

        PlanetF = enhance_class(
            Planet,
            PlanetCatalogueFile,
            catalogue_file=str(csv_path),
        )
        assert PlanetF.fullRadius == pytest.approx(1.2 * 71492000, rel=1e-4)
        assert PlanetF.fullMass == pytest.approx(1.5 * 1.898e27, rel=1e-4)
        assert PlanetF._orbit_period == 3.5
        assert PlanetF._inclination == 87.0
        assert PlanetF._eccentricity == 0.01
        assert PlanetF._transit_time == 7200

    def test_defaults_when_no_file(self):
        """Defaults are used when no catalogue_file is given."""
        PlanetF = enhance_class(Planet, PlanetCatalogueFile)
        assert PlanetF._orbit_period == 2.0
        assert PlanetF._inclination == 90.0
        assert PlanetF._eccentricity == 0.0


# ---------------------------------------------------------------------------
#  StarCatalogueFile tests
# ---------------------------------------------------------------------------


class TestStarCatalogueFile:
    """Tests for the file-based star catalogue mixin."""

    def test_populates_from_csv(self, tmp_path):
        """Star parameters are loaded from a CSV catalogue file."""
        csv_path = tmp_path / "stars.csv"
        _write_csv(
            csv_path,
            [
                "star_temperature [K]",
                "star_radius [R_sun]",
                "star_mass [M_sun]",
                "star_metallicity [dex]",
                "star_distance [pc]",
            ],
            [[6200, 1.3, 1.1, 0.1, 50.0]],
        )

        StarF = enhance_class(
            BlackbodyStar,
            StarCatalogueFile,
            catalogue_file=str(csv_path),
        )
        assert StarF.temperature == 6200
        assert StarF.radius == pytest.approx(1.3 * 6.957e8, rel=1e-4)
        assert StarF._metallicity == 0.1
        assert StarF.distance == 50.0

    def test_defaults_when_no_file(self):
        """Defaults are used when no catalogue_file is given."""
        StarF = enhance_class(BlackbodyStar, StarCatalogueFile)
        assert StarF.temperature == 5000
        assert StarF._metallicity == 0.0


# ---------------------------------------------------------------------------
#  PlanetCatalogueExomast tests (mocked HTTP)
# ---------------------------------------------------------------------------


class TestPlanetCatalogueExomast:
    """Tests for the ExoMAST planet catalogue mixin (mocked)."""

    @patch("taurex.mixin._catalogue_planet.requests.get")
    def test_populates_from_exomast(self, mock_get):
        """Planet parameters are fetched from ExoMAST API."""
        mock_get.return_value.json.return_value = {
            "pl_masse": [0.5],
            "pl_masse_unit": "Mj",
            "pl_rade": [1.1],
            "pl_rade_unit": "Rj",
            "pl_orbsmax": [0.03],
            "pl_orbper": [1.5],
            "pl_orbincl": [85.0],
            "pl_orbeccen": [0.0],
            "pl_trandur": [5400],
        }
        mock_get.return_value.raise_for_status = lambda: None

        PlanetE = enhance_class(
            Planet,
            PlanetCatalogueExomast,
            planet_name="WASP-12 b",
        )
        assert PlanetE.fullMass == pytest.approx(0.5 * 1.898e27, rel=1e-4)
        assert PlanetE.fullRadius == pytest.approx(1.1 * 71492000, rel=1e-4)
        assert PlanetE._orbit_period == 1.5
        assert PlanetE._inclination == 85.0
        assert PlanetE._transit_time == 5400

    @patch("taurex.mixin._catalogue_planet.requests.get")
    def test_defaults_on_http_error(self, mock_get):
        """Defaults are used when the ExoMAST API call fails."""
        import requests as req

        mock_get.side_effect = req.RequestException("network unreachable")

        PlanetE = enhance_class(
            Planet,
            PlanetCatalogueExomast,
            planet_name="Nonexistent b",
        )
        assert PlanetE._orbit_period == 2.0
        assert PlanetE._inclination == 90.0

    def test_defaults_when_no_planet_name(self):
        """Defaults are used when no planet_name is given."""
        PlanetE = enhance_class(Planet, PlanetCatalogueExomast)
        assert PlanetE._orbit_period == 2.0
        assert PlanetE._inclination == 90.0


# ---------------------------------------------------------------------------
#  StarCatalogueExomast tests (mocked HTTP)
# ---------------------------------------------------------------------------


class TestStarCatalogueExomast:
    """Tests for the ExoMAST star catalogue mixin (mocked)."""

    @patch("taurex.mixin._catalogue_star.requests.get")
    def test_populates_from_exomast(self, mock_get):
        """Star parameters are fetched from ExoMAST API."""
        mock_get.return_value.json.return_value = {
            "st_teff": [7200],
            "st_rad": [1.5],
            "st_mass": [1.3],
            "st_met": [-0.2],
            "sy_dist": [120.0],
        }
        mock_get.return_value.raise_for_status = lambda: None

        StarE = enhance_class(
            BlackbodyStar,
            StarCatalogueExomast,
            planet_name="HD 209458 b",
        )
        assert StarE.temperature == 7200
        assert StarE.radius == pytest.approx(1.5 * 6.957e8, rel=1e-4)
        assert StarE._metallicity == -0.2
        assert StarE.distance == 120.0

    def test_defaults_when_no_planet_name(self):
        """Defaults are used when no planet_name is given."""
        StarE = enhance_class(BlackbodyStar, StarCatalogueExomast)
        assert StarE.temperature == 5000
        assert StarE._metallicity == 0.0

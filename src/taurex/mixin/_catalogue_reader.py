"""Reader classes for CSV/TSV catalogue files."""

import re

import pandas as pd

from taurex.log import Logger


class CatalogueReader(Logger):
    """Base class for catalogue readers."""

    def __init__(self) -> None:
        """Initialise the catalogue reader."""
        super().__init__(self.__class__.__name__)

    def load_planet_list(self):
        """Load the list of available planets (optional)."""
        raise NotImplementedError


class FileReader(CatalogueReader):
    """Parses a CSV catalogue file and returns star/planet parameters.

    Parameters
    ----------
    filename : str
        Path to the CSV file.
    target_no : int, optional
        Row index to read (default 0).
    target_name : str or None, optional
        Planet name to look up (overrides *target_no*).
    """

    def __init__(self, filename=None, target_no=0, target_name=None):
        """Initialise with file path and target selection."""
        super().__init__()
        self.filename = filename
        self.target_no = target_no
        self.target_name = target_name
        self.star_params, self.planet_params = self.load_target_list(self.filename)

    def load_target_list(self, filename):
        """Parse the CSV and return (star_params, planet_params) lists."""
        df = pd.read_csv(filename)

        star_values, star_field, star_units = [], [], []
        planet_values, planet_field, planet_units = [], [], []

        for col in df.columns:
            if (
                self.target_name is None
                and df.index.get_loc(df.index[self.target_no]) is not None
            ):
                row_idx = self.target_no
            elif self.target_name is not None:
                mask = (
                    df[col]
                    .astype(str)
                    .str.contains(self.target_name, case=False, na=False)
                )
                row_idx = df[mask].index[0] if mask.any() else self.target_no
            else:
                row_idx = self.target_no

            value = df[col].iloc[row_idx]
            col_lower = col.lower()

            if col_lower.startswith("star"):
                star_field.append(col)
                star_values.append(value)
                unit_match = re.search(r"\[(.+?)\]", col)
                star_units.append(unit_match.group(1) if unit_match else None)
            elif col_lower.startswith("planet"):
                planet_field.append(col)
                planet_values.append(value)
                unit_match = re.search(r"\[(.+?)\]", col)
                planet_units.append(unit_match.group(1) if unit_match else None)

        star_params = list(zip(star_field, star_values, star_units, strict=False))
        planet_params = list(
            zip(planet_field, planet_values, planet_units, strict=False)
        )
        return star_params, planet_params

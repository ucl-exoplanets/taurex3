"""Chemistry from file module."""
import typing as t

import numpy as np
import numpy.typing as npt

from taurex.output import OutputGroup
from taurex.types import PathLike

from .autochemistry import AutoChemistry


class ChemistryFile(AutoChemistry):
    """Chemistry profile read from file."""

    def __init__(self, gases: t.List[str] = None, filename: PathLike = None):
        """Initialize chemistry from file.


        Parameters
        ----------

        gases : :obj:`list`
            Gases in file

        filename : PathLike
            filename for mix ratios

        """
        super().__init__(self.__class__.__name__)

        self._gases = gases
        self._filename = filename
        self._mix_ratios = np.loadtxt(filename).T
        self.determine_active_inactive()

    @property
    def gases(self) -> t.List[str]:
        """Gases in file."""
        return self._gases

    @property
    def mixProfile(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Mix ratios."""
        return self._mix_ratios

    def write(self, output: OutputGroup) -> OutputGroup:
        """Write chemistry to output group."""
        gas_entry = super().write(output)
        gas_entry.write_scalar("filename", self._filename)

        return gas_entry

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Input keywords for chemistry from file."""
        return (
            "file",
            "fromfile",
        )

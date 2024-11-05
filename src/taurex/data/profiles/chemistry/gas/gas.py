"""Base Gas class."""
import typing as t

import numpy as np
import numpy.typing as npt

from taurex.data.citation import Citable
from taurex.data.fittable import Fittable
from taurex.log import Logger
from taurex.output import OutputGroup
from taurex.output.writeable import Writeable


class Gas(Fittable, Logger, Writeable, Citable):
    """This class is a base for a single molecule or gas.
    Its used to describe how it mixes at each layer and combined
    with
    :class:`~taurex.data.profile.chemistry.taurexchemistry.TaurexChemistry`
    is used to build a chemical profile of the planets atmosphere.
    Requires implementation of:

    - func:`~mixProfile`

    """

    def __init__(self, name: str, molecule_name: str) -> None:
        """Initialize base gas class.

        Parameters
        ----------
        name: str
            Identifier for class.
        molecule_name: str
            Name of molecule


        """
        Logger.__init__(self, name)
        Fittable.__init__(self)
        self._molecule_name = molecule_name
        self.mix_profile = None

    @property
    def molecule(self) -> str:
        """
        Returns
        -------
        molecule_name: str
            Name of molecule
        """
        return self._molecule_name

    @property
    def mixProfile(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Return mix profile for each layer.
        **Requires implementation**

        Should return mix profile of molecule/gas at each layer

        Returns
        -------
        mix: :obj:`array`
            Mix ratio for molecule at each layer

        """
        raise NotImplementedError

    def initialize_profile(
        self,
        nlayers: t.Optional[int] = None,
        temperature_profile: t.Optional[npt.NDArray[np.float64]] = None,
        pressure_profile: t.Optional[npt.NDArray[np.float64]] = None,
        altitude_profile: t.Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        """Initializes and computes mix profile.

        Parameters
        ----------
        nlayers: int
            Number of layers in atmosphere

        temperature_profile: :obj:`array`
            Temperature profile of atmosphere in K. Length must be
            equal to ``nlayers``

        pressure_profile: :obj:`array`
            Pressure profile of atmosphere in Pa. Length must be
            equal to ``nlayers``

        altitude_profile: :obj:`array`
            Altitude profile of atmosphere in m. Length must be
            equal to ``nlayers``

        """
        pass

    def write(self, output: OutputGroup) -> OutputGroup:
        """
        Writes class and arguments to file

        Parameters
        ----------
        output: :class:`~taurex.output.output.Output`

        """
        gas_entry = output.create_group(self.molecule)
        gas_entry.write_string("gas_type", self.__class__.__name__)
        gas_entry.write_string("molecule_name", self._molecule_name)
        return gas_entry

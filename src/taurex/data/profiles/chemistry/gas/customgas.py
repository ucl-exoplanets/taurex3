"""Constant gas profile."""
import typing as t

import numpy as np
import numpy.typing as npt

from taurex.output import OutputGroup
from taurex.util import molecule_texlabel

from .gas import Gas


class CustomGas(Gas):
    """Constant gas profile.

    Molecular abundace is constant at each layer of the
    atmosphere

    """

    def __init__(
        self,
        molecule_name: t.Optional[str] = "H2O",
        mix_ratio: npt.NDArray[np.float64] = None,
    ) -> None:
        """Initialize constant gas profile.

        Parameters
        -----------
        molecule_name : str
            Name of molecule

        mix_ratio : float
            Mixing ratio of the molecule


        """
        super().__init__("CustomGas", molecule_name)
        self._mix_ratio = mix_ratio
        self.add_active_gas_param()

    @property
    def mixProfile(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """

        Mixing profile

        Returns
        -------
        mix: :obj:`array`
            Mix ratio for molecule at each layer

        """

        return self._mix_array

    def initialize_profile(
        self,
        nlayers: t.Optional[int] = None,
        temperature_profile: t.Optional[npt.NDArray[np.float64]] = None,
        pressure_profile: t.Optional[npt.NDArray[np.float64]] = None,
        altitude_profile: t.Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        """Initialize the mixing profile.

        Parameters
        -----------
        nlayers: int
            Number of layers in atmosphere
        temperature_profile: :obj:`array`
            Temperature profile of atmosphere
        pressure_profile: :obj:`array`
            Pressure profile of atmosphere
        altitude_profile: :obj:`array`
            Altitude profile of atmosphere, deprecated

        """
        self._mix_array = self._mix_ratio

    def add_active_gas_param(self) -> None:
        """Add the mixing ratio as a fitting parameter.

        Fitting parameter identifier is the molecule name.
        Generates a fitting parameter on the fly by building
        getter and setter functions and passing them to
        :func:`taurex.fitting.fittable.Fittable.add_fittable_param`

        """

        mol_name = self.molecule
        param_name = self.molecule
        param_tex = molecule_texlabel(mol_name)

        def read_mol(self):
            return self._mix_ratio

        def write_mol(self, value):
            self._mix_ratio = value

        read_mol.__doc__ = f"{mol_name} constant mix ratio (VMR)"

        fget = read_mol
        fset = write_mol

        bounds = [1.0e-12, 0.1]

        default_fit = False
        self.add_fittable_param(
            param_name, param_tex, fget, fset, "log", default_fit, bounds
        )

    def write(self, output: OutputGroup):
        """Write constant gas profile to output."""
        gas_entry = super().write(output)
        gas_entry.write_scalar("mix_ratio", self._mix_ratio)

        return gas_entry

    @classmethod
    def input_keywords(cls) -> t.Tuple[str]:
        return ("custom",)

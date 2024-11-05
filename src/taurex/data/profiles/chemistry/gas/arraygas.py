"""Gas from an array."""
import typing as t

import numpy as np
import numpy.typing as npt

from taurex.output import OutputGroup

from .gas import Gas


class ArrayGas(Gas):
    """Gas profile from an array.

    Molecular abundance is interpolated if the number of layers do not match

    """

    def __init__(
        self,
        molecule_name: t.Optional[str] = "H2O",
        mix_ratio_array: t.Optional[t.List[float]] = None,
    ) -> None:
        super().__init__(self.__class__.__name__, molecule_name)
        """Initialize gas profile from an array.

        Parameters
        -----------
        molecule_name : str
            Name of molecule

        mix_ratio_array : :obj:`array`
            Mixing ratio of the molecule at each layer


        """
        mix_ratio_array = mix_ratio_array or [1e-2, 1e-6]
        self._mix_ratio_array = np.array(mix_ratio_array)

    @property
    def mixProfile(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Mixing profile.

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
        """Initialize the gas profile.

        Parameters
        -----------
        nlayers : int
            Number of layers in the atmosphere
        temperature_profile : :obj:`array`
            Temperature profile of the atmosphere
        pressure_profile : :obj:`array`
            Pressure profile of the atmosphere
        altitude_profile : :obj:`array`, optional
            Altitude profile of the atmosphere (Deprecated)

        """
        if nlayers is None:
            self.error("number layers argument required")
            raise ValueError("Number of layers argument required")

        interp_array = np.linspace(0.0, 1.0, self._mix_ratio_array.shape[0])

        layer_interp = np.linspace(0.0, 1.0, nlayers)

        self._mix_array = np.interp(layer_interp, interp_array, self._mix_ratio_array)

    def write(self, output: OutputGroup) -> OutputGroup:
        """Write parameters to output.


        Parameters
        ----------
        output: :class:`~taurex.output.output.Output`
        """
        gas_entry = super().write(output)
        gas_entry.write_array("mix_ratio_array", self._mix_ratio_array)

        return gas_entry

    @classmethod
    def input_keywords(cls) -> t.Tuple[str]:
        """Input keywords for this class."""
        return (
            "array",
            "fromarray",
        )

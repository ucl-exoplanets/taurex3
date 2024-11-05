"""Two point gas profile."""
import math
import typing as t

import numpy as np
import numpy.typing as npt

from taurex.output import OutputGroup
from taurex.util import molecule_texlabel

from .gas import Gas


class TwoPointGas(Gas):
    """Two point gas profile.

    A gas profile with two different mixing layers at the surface of the
    planet and top of the atmosphere and interpolated between the two


    """

    def __init__(
        self,
        molecule_name: t.Optional[str] = "CH4",
        mix_ratio_surface: t.Optional[float] = 1e-4,
        mix_ratio_top: t.Optional[float] = 1e-8,
    ):
        """Initialize a two point gas profile.

        Parameters
        -----------
        molecule_name : str
            Name of molecule

        mix_ratio_surface : float
            Mixing ratio of the molecule on the planet surface

        mix_ratio_top : float
            Mixing ratio of the molecule at the top of the atmosphere

        """
        super().__init__("TwoPointGas", molecule_name)
        self._mix_surface = mix_ratio_surface
        self._mix_top = mix_ratio_top
        self.add_surface_param()
        self.add_top_param()
        self._mix_profile = None

    @property
    def mixProfile(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Mixing ratio profile.

        Returns
        -------
        mix: :obj:`array`
            Mix ratio for molecule at each layer
        """
        return self._mix_profile

    @property
    def mixRatioSurface(self) -> float:  # noqa: N802
        """Abundance on the planets surface"""
        return self._mix_surface

    @property
    def mixRatioTop(self) -> float:  # noqa: N802
        """Abundance on the top of atmosphere"""
        return self._mix_top

    @mixRatioSurface.setter
    def mixRatioSurface(self, value: float) -> None:  # noqa: N802
        """Set abundance on the planets surface"""
        self._mix_surface = value

    @mixRatioTop.setter
    def mixRatioTop(self, value: float) -> None:  # noqa: N802
        """Set abundance on the top of atmosphere"""
        self._mix_top = value

    def add_surface_param(self) -> None:
        """Add surface parameter.

        Adds fittable parameter for the surface abundance of the molecule
        with format ``[molecule]_surface``

        """
        param_name = self.molecule
        param_tex = molecule_texlabel(param_name)

        param_surface = f"{param_name}_surface"
        param_surf_tex = f"{param_tex}_surface"

        def read_surf(self):
            return self._mix_surface

        def write_surf(self, value):
            self._mix_surface = value

        read_surf.__doc__ = f"Abundance of {param_name} on the planets surface"

        fget_surf = read_surf
        fset_surf = write_surf

        bounds = [1.0e-12, 0.1]

        default_fit = False
        self.add_fittable_param(
            param_surface,
            param_surf_tex,
            fget_surf,
            fset_surf,
            "log",
            default_fit,
            bounds,
        )

    def add_top_param(self):
        """Add top parameter.

        Adds fittable parameter for the top abundance of the molecule
        with format ``[molecule]_top``

        """
        param_name = self.molecule
        param_tex = molecule_texlabel(param_name)

        param_top = f"{param_name}_top"
        param_top_tex = f"{param_tex}_top"

        def read_top(self):
            return self._mix_top

        def write_top(self, value):
            self._mix_top = value

        read_top.__doc__ = f"Abundance of {param_name} on the top of atmosphere"

        fget_top = read_top
        fset_top = write_top

        bounds = [1.0e-12, 0.1]

        default_fit = False
        self.add_fittable_param(
            param_top, param_top_tex, fget_top, fset_top, "log", default_fit, bounds
        )

    def initialize_profile(
        self,
        nlayers: t.Optional[int] = None,
        temperature_profile: t.Optional[npt.NDArray[np.float64]] = None,
        pressure_profile: t.Optional[npt.NDArray[np.float64]] = None,
        altitude_profile: t.Optional[npt.NDArray[np.float64]] = None,
    ):
        """Initialize the profile.

        Parameters
        -----------
        nlayers : int
            number of layers
        temperature_profile : np.ndarray
            temperature profile
        pressure_profile : np.ndarray
            pressure profile
        altitude_profile : np.ndarray , optional
            altitude profile, deprecated

        """
        self._mix_profile = np.zeros(nlayers)

        chem_surf = self._mix_surface
        chem_top = self._mix_top
        p_surf = pressure_profile[0]
        p_top = pressure_profile[-1]

        a = (math.log10(chem_surf) - math.log10(chem_top)) / (
            math.log10(p_surf) - math.log10(p_top)
        )

        b = math.log10(chem_surf) - a * math.log10(p_surf)

        self._mix_profile[1:-1] = 10 ** (a * np.log10(pressure_profile[1:-1]) + b)
        self._mix_profile[0] = chem_surf
        self._mix_profile[-1] = chem_top

    def write(self, output: OutputGroup) -> OutputGroup:
        gas_entry = super().write(output)
        gas_entry.write_scalar("mix_ratio_top", self.mixRatioTop)
        gas_entry.write_scalar("mix_ratio_surface", self.mixRatioSurface)
        return gas_entry

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        return (
            "twopoint",
            "2point",
        )

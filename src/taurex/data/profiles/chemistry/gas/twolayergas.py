"""Two layer gas profile."""

import typing as t

import numpy as np
import numpy.typing as npt

from taurex.output import OutputGroup
from taurex.util import molecule_texlabel, movingaverage

from .gas import Gas


class TwoLayerGas(Gas):
    """Two layer gas profile.

    A gas profile with two different mixing layers at the surface of the
    planet and top of the atmosphere seperated at a defined
    pressure point and smoothened.

    """

    def __init__(
        self,
        molecule_name: t.Optional[str] = "CH4",
        mix_ratio_surface: t.Optional[float] = 1e-4,
        mix_ratio_top: t.Optional[float] = 1e-8,
        mix_ratio_P: t.Optional[float] = 1e3,  # noqa: N803
        mix_ratio_smoothing: t.Optional[float] = 10,
    ):
        """Initialize a two layer gas profile.

        Parameters
        -----------
        molecule_name : str
        Name of molecule

        mix_ratio_surface : float
            Mixing ratio of the molecule on the planet surface

        mix_ratio_top : float
            Mixing ratio of the molecule at the top of the atmosphere

        mix_ratio_P : float
            Boundary Pressure point between the two layers

        mix_ratio_smoothing : float , optional
            smoothing window

        """
        super().__init__(self.__class__.__name__, molecule_name=molecule_name)

        if mix_ratio_smoothing <= 0:
            raise ValueError("Smoothing window must be positive")

        self._mix_surface = mix_ratio_surface or 1e-4
        self._mix_top = mix_ratio_top
        self._mix_ratio_pressure = mix_ratio_P
        self._mix_ratio_smoothing = mix_ratio_smoothing
        self._mix_profile = None
        self.add_surface_param()
        self.add_top_param()
        self.add_pressure_param()

    @property
    def mixProfile(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Mixing ratio profile for molecule.

        Returns
        -------
        mix: :obj:`array`
            Mix ratio for molecule at each layer

        """
        return self._mix_profile

    @property
    def mixRatioSurface(self) -> float:  # noqa: N802
        """Abundance on the planets surface."""
        return self._mix_surface

    @property
    def mixRatioTop(self) -> float:  # noqa: N802
        """Abundance on the top of atmosphere."""
        return self._mix_top

    @property
    def mixRatioPressure(self) -> float:  # noqa: N802
        """Pressure at which the abundance changes."""
        return self._mix_ratio_pressure

    @property
    def mixRatioSmoothing(self) -> float:  # noqa: N802
        """Smoothing window."""
        return self._mix_ratio_smoothing

    @mixRatioSurface.setter
    def mixRatioSurface(self, value: float) -> None:  # noqa: N802
        """Set abundance on the planets surface."""
        self._mix_surface = value

    @mixRatioTop.setter
    def mixRatioTop(self, value: float) -> None:  # noqa: N802
        """Set abundance on the top of atmosphere."""
        self._mix_top = value

    @mixRatioPressure.setter
    def mixRatioPressure(self, value: float) -> None:  # noqa: N802
        """Set pressure at which the abundance changes."""
        self._mix_pressure = value

    @mixRatioSmoothing.setter
    def mixRatioSmoothing(self, value: float) -> None:  # noqa: N802
        """Set smoothing window."""
        self._mix_smoothing = value

    def add_surface_param(self) -> None:
        """Generates surface fitting parameters.

        Has the form ``[Molecule]_surface``
        """
        param_name = self.molecule
        param_tex = molecule_texlabel(param_name)

        param_surface = f"{param_name}_surface"
        param_surf_tex = f"{param_tex}_surface"

        def read_surf(self):
            return self._mix_surface

        def write_surf(self, value):
            self._mix_surface = value

        read_surf.__doc__ = (
            f"Mixing ratio at the surface of the planet for {param_name}"
        )

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

    def add_top_param(self) -> None:
        """Generates TOA fitting parameters.

        Has the form: ``[Molecule]_top``
        """
        param_name = self.molecule
        param_tex = molecule_texlabel(param_name)

        param_top = f"{param_name}_top"
        param_top_tex = f"{param_tex}_top"

        def read_top(self):
            return self._mix_top

        def write_top(self, value):
            self._mix_top = value

        read_top.__doc__ = f"Mixing ratio at the top of the atmosphere for {param_name}"

        fget_top = read_top
        fset_top = write_top

        bounds = [1.0e-12, 0.1]

        default_fit = False
        self.add_fittable_param(
            param_top, param_top_tex, fget_top, fset_top, "log", default_fit, bounds
        )

    def add_pressure_param(self) -> None:
        """Generates pressure fitting parameter.

        Has the form ``[Molecule]_P``
        """
        mol_name = self.molecule
        mol_tex = molecule_texlabel(mol_name)

        param_p = f"{mol_name}_P"
        param_p_tex = f"{mol_tex}_P"

        def read_p(self: "TwoLayerGas"):
            return self._mix_ratio_pressure

        def write_p(self: "TwoLayerGas", value: float):
            self._mix_ratio_pressure = value

        read_p.__doc__ = f"Pressure at which the mixing ratio changes for {mol_name}"

        fget_p = read_p
        fset_p = write_p

        bounds = [1.0e-12, 0.1]

        default_fit = False
        self.add_fittable_param(
            param_p, param_p_tex, fget_p, fset_p, "log", default_fit, bounds
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

        smooth_window = self._mix_ratio_smoothing
        p_layer = np.abs(pressure_profile - self._mix_ratio_pressure).argmin()

        start_layer = max(int(p_layer - smooth_window / 2), 0)

        end_layer = min(int(p_layer + smooth_window / 2), nlayers - 1)

        p_nodes = [
            pressure_profile[0],
            pressure_profile[start_layer],
            pressure_profile[end_layer],
            pressure_profile[-1],
        ]

        c_nodes = [
            self.mixRatioSurface,
            self.mixRatioSurface,
            self.mixRatioTop,
            self.mixRatioTop,
        ]

        chemprofile = 10 ** np.interp(
            (np.log(pressure_profile[::-1])),
            np.log(p_nodes[::-1]),
            np.log10(c_nodes[::-1]),
        )

        wsize = nlayers * (smooth_window / 100.0)

        if wsize % 2 == 0:
            wsize += 1

        c_smooth = 10 ** movingaverage(np.log10(chemprofile), int(wsize))

        border = np.int((len(chemprofile) - len(c_smooth)) / 2)

        self._mix_profile = chemprofile[::-1]

        self._mix_profile[border:-border] = c_smooth[::-1]

    def write(self, output: OutputGroup):
        """Write gas profile to output."""
        gas_entry = super().write(output)
        gas_entry.write_scalar("mix_ratio_top", self.mixRatioTop)
        gas_entry.write_scalar("mix_ratio_surface", self.mixRatioSurface)
        gas_entry.write_scalar("mix_ratio_P", self.mixRatioPressure)
        gas_entry.write_scalar("mix_ratio_smoothing", self.mixRatioSmoothing)

        return gas_entry

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, str]:
        return (
            "twolayer",
            "2layer",
        )

    BIBTEX_ENTRIES = [
        """
        @misc{changeat2019complex,
            title={Towards a more complex description of chemical profiles
            in exoplanets retrievals: A 2-layer parameterisation},
            author={Quentin Changeat and Billy Edwards and
                Ingo Waldmann and Giovanna Tinetti},
            year={2019},
            eprint={1903.11180},
            archivePrefix={arXiv},
            primaryClass={astro-ph.EP}
        }
        """,
    ]

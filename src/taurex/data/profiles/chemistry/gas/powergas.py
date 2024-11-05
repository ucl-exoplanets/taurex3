"""Gas profile in Power law."""

import typing as t

import numpy as np
import numpy.typing as npt

from taurex.output import OutputGroup
from taurex.util import molecule_texlabel

from .gas import Gas

ProfileType = t.Literal["auto", "H2", "H2O", "TiO", "VO", "H-", "Na", "K"]
"""Profile type for power law."""

KNOWN_MOLECULES = ["H2", "H2O", "TiO", "VO", "H-", "Na", "K"]
"""Known molecules for power law."""


class PowerGas(Gas):
    """Gas profile in Power law.

    This is a profile adapted for HJ (T > 2000 K) which takes
    into account upper atm mix reduction. Laws taken from Parmentier (2018)

    """

    def __init__(
        self,
        molecule_name: t.Optional[str] = "H2O",
        profile_type: ProfileType = "auto",
        mix_ratio_surface: t.Optional[float] = None,
        alpha: t.Optional[float] = None,
        beta: t.Optional[float] = None,
        gamma: t.Optional[float] = None,
    ):
        """Initialize power gas profile.

        Parameters
        -----------
        molecule_name : str
            Name of molecule (for the xsec)

        profile_type : str , optional
            name of the molecule to take the profile from: 'H2','H2O','TiO',
                'VO', 'H-', 'Na', 'K'
            by default it uses 'auto' to get the same profile as for molecule_name

        mix_ratio_surface : float , optional
            Mixing ratio of the molecule on the planet surface

        alpha : float , optional
            pressure dependance coefficient approx 10^alpha

        beta : float , optional
            temperature dependance coefficient approx 10^(beta/T)

        gamma : float , optional
            scale coefficient

        Raises
        -------
        ValueError
            if the molecule is not known and profile_type is not ``auto``

        """
        super().__init__("PowerGas", molecule_name=molecule_name)
        if profile_type == "auto":
            self._profile_type = molecule_name
        elif profile_type not in KNOWN_MOLECULES:
            raise ValueError(
                f"Unknown profile type {profile_type}, "
                f"please choose from {KNOWN_MOLECULES} if not auto"
            )
        else:
            self._profile_type = profile_type

        self._mix_surface = mix_ratio_surface
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._mix_profile = None
        self.add_surface_param()
        self.add_alpha_param()
        self.add_beta_param()
        self.add_gamma_param()

    @property
    def mixProfile(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Mixing ratio profile."""
        return self._mix_profile

    @property
    def mixRatioSurface(self) -> float:  # noqa: N802
        """Abundance on the planets surface."""
        return self._mix_surface

    @property
    def alpha(self) -> float:
        """Pressure dependance coefficient."""
        return self._alpha

    @property
    def beta(self) -> float:
        """Temperature dependance coefficient."""
        return self._beta

    @property
    def gamma(self) -> float:
        """Scale coefficient."""
        return self._gamma

    @mixRatioSurface.setter
    def mixRatioSurface(self, value: float):  # noqa: N802
        """Set the abundance on the planets surface."""
        self._mix_surface = value

    @alpha.setter
    def alpha(self, value: float):
        """Set the pressure dependance coefficient."""
        self._alpha = value

    @beta.setter
    def beta(self, value: float):
        """Set the temperature dependance coefficient."""
        self._beta = value

    @gamma.setter
    def gamma(self, value: float):
        """Set the scale coefficient."""
        self._gamma = value

    def add_surface_param(self):
        """Add surface parameter to fit."""
        param_name = self.molecule
        param_tex = molecule_texlabel(param_name)

        param_surface = f"{param_name}_surface"
        param_surf_tex = f"{param_tex}_surface"

        def read_surf(self):
            return self._mix_surface

        def write_surf(self, value):
            self._mix_surface = value

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

    def add_alpha_param(self):
        """Add alpha parameter to fit."""
        param_name = self.molecule
        param_tex = molecule_texlabel(param_name)

        param_alpha = f"{param_name}_alpha"
        param_alpha_tex = f"{param_tex}_alpha"

        def read_alpha(self):
            return self._alpha

        def write_alpha(self, value):
            self._alpha = value

        fget_alpha = read_alpha
        fset_alpha = write_alpha

        bounds = [0.5, 2.5]

        default_fit = False
        self.add_fittable_param(
            param_alpha,
            param_alpha_tex,
            fget_alpha,
            fset_alpha,
            "linear",
            default_fit,
            bounds,
        )

    def add_beta_param(self):
        """Add beta parameter to fit."""
        param_name = self.molecule
        param_tex = molecule_texlabel(param_name)

        param_beta = f"{param_name}_beta"
        param_beta_tex = f"{param_tex}_beta"

        def read_beta(self):
            return self._beta

        def write_beta(self, value):
            self._beta = value

        fget_beta = read_beta
        fset_beta = write_beta

        bounds = [1e4, 6e4]

        default_fit = False
        self.add_fittable_param(
            param_beta, param_beta_tex, fget_beta, fset_beta, "log", default_fit, bounds
        )

    def add_gamma_param(self):
        """Add gamma parameter to fit."""
        param_name = self.molecule
        param_tex = molecule_texlabel(param_name)

        param_gamma = f"{param_name}_gamma"
        param_gamma_tex = f"{param_tex}_gamma"

        def read_gamma(self):
            return self._gamma

        def write_gamma(self, value):
            self._gamma = value

        fget_gamma = read_gamma
        fset_gamma = write_gamma

        bounds = [5, 25]

        default_fit = False
        self.add_fittable_param(
            param_gamma,
            param_gamma_tex,
            fget_gamma,
            fset_gamma,
            "linear",
            default_fit,
            bounds,
        )

    def check_known(
        self,
        molecule_name: t.Literal["H2", "H2O", "TiO", "VO", "H-", "Na", "K"],
    ) -> t.Tuple[float, float, float, float]:
        """Check if the molecule is known and return the coefficients.

        Parameters
        -----------
        molecule_name : str
            name of the molecule to take the profile from: 'H2','H2O','TiO',
                'VO', 'H-', 'Na', 'K'

        Returns
        --------
        a_coeff : float
            power coefficient for pressure or None
        b_coeff : float
            power coefficient for temperature or None
        g_coeff : float
            scale coefficient or None
        big_a_coeff : float
            mixing ratio at the surface or None

        Raises
        -------
        ValueError
            if the molecule is not known

        """
        a_coeff = np.array([1.0, 2.0, 1.6, 1.5, 0.6, 0.6, 0.6])
        b_coeff = np.array([2.41, 4.83, 5.94, 5.4, -0.14, 1.89, 1.28]) * 1e-4
        g_coeff = np.array([6.5, 15.9, 23.0, 23.8, 7.7, 12.2, 12.7])
        big_a_coeff = np.power(10, np.array([-0.1, -3.3, -7.1, -9.2, -8.3, -5.5, -7.1]))
        if molecule_name in KNOWN_MOLECULES:
            i = KNOWN_MOLECULES.index(molecule_name)
            self.debug(
                "%s %s %s %s %s", i, a_coeff[i], b_coeff[i], g_coeff[i], big_a_coeff[i]
            )
            return a_coeff[i], b_coeff[i], g_coeff[i], big_a_coeff[i]
        else:
            raise ValueError(
                f"molecule {molecule_name} is not known, "
                f"please choose from {KNOWN_MOLECULES}"
            )

    def initialize_profile(
        self,
        nlayers: int,
        temperature_profile: npt.NDArray[np.float64],
        pressure_profile: npt.NDArray[np.float64],
        altitude_profile: t.Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
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
        molecule_name = self._profile_type
        coeffs = None, None, None, None
        if molecule_name in KNOWN_MOLECULES:
            coeffs = self.check_known(molecule_name=molecule_name)

        mix_surface = self._mix_surface
        alpha = self._alpha
        beta = self._beta
        gamma = self._gamma
        ad = []

        if self._mix_surface is None:
            if coeffs[3] is not None:
                mix_surface = coeffs[3]
            else:
                self.error("molecule %s has a missing power coefficient", molecule_name)
                raise ValueError
        if self._alpha is None:
            if coeffs[0] is not None:
                alpha = coeffs[0]
            else:
                self.error("molecule %s has a missing power coefficient", molecule_name)
                raise ValueError
        if self._beta is None:
            if coeffs[1] is not None:
                beta = coeffs[1]
            else:
                self.error("molecule %s has a missing power coefficient", molecule_name)
                raise ValueError
        if self._gamma is None:
            if coeffs[2] is not None:
                gamma = coeffs[2]
            else:
                self.error("molecule %s has a missing power coefficient", molecule_name)
                raise ValueError
        # print('coeffs: ', coeffs)
        # print('a, b, g, minx_S: ', alpha, beta, gamma, mix_surface)

        # Ad = alpha * np.log10(pressure_profile) + beta / temperature_profile + gamma

        pressure = pressure_profile * 1e-5  # convert pressure to bar
        ad = (
            np.power(10, gamma * (-1))
            * np.power(pressure, alpha)
            * np.power(10, beta / temperature_profile)
        )
        mix = 1 / np.sqrt(mix_surface) + 1 / np.sqrt(ad)
        mix = np.power(1 / mix, 2)

        self._mix_profile = mix

    def write(self, output: OutputGroup) -> OutputGroup:
        """Write the gas profile to output."""
        gas_entry = super().write(output)
        gas_entry.write_scalar("alpha", self.alpha)
        gas_entry.write_scalar("mix_ratio_surface", self.mixRatioSurface)
        gas_entry.write_scalar("beta", self.beta)
        gas_entry.write_scalar("gamma", self.gamma)

        return gas_entry

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, str, str]:
        """Return input keywords for power law gas profile."""
        return (
            "power",
            "powerchemistry",
            "parmentier",
        )

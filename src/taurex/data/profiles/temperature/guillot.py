"""Guillot 2010 temperature profile."""
import typing as t

import numpy as np
import numpy.typing as npt

from taurex.data.fittable import fitparam
from taurex.exceptions import InvalidModelException
from taurex.output import OutputGroup

from .tprofile import TemperatureProfile


class Guillot2010(TemperatureProfile):
    """TP profile from Guillot 2010, A&A, 520, A27 (equation 49).

    Using modified 2stream approx.
    from Line et al. 2012, ApJ, 749,93 (equation 19)

    """

    def __init__(
        self,
        T_irr: t.Optional[float] = 1500,  # noqa: N803
        kappa_irr: t.Optional[float] = 0.01,
        kappa_v1: t.Optional[float] = 0.005,
        kappa_v2: t.Optional[float] = 0.005,
        alpha: t.Optional[float] = 0.5,
        T_int: t.Optional[float] = 100,  # noqa: N803
    ):
        """Initialize guillot temperature profile.

        Parameters
        -----------
            T_irr: float
                planet equilibrium temperature
                (Line fixes this but we keep as free parameter)
            kappa_ir: float
                mean infra-red opacity
            kappa_v1: float
                mean optical opacity one
            kappa_v2: float
                mean optical opacity two
            alpha: float
                ratio between kappa_v1 and kappa_v2 downwards radiation stream
            T_int: float
                Internal heating parameter (K)

        """
        super().__init__("Guillot")

        self.T_irr = T_irr

        self.kappa_ir = kappa_irr
        self.kappa_v1 = kappa_v1
        self.kappa_v2 = kappa_v2
        self.alpha = alpha
        self.T_int = T_int

        self._check_values()

    @fitparam(
        param_name="T_irr",
        param_latex=r"$T_\\mathrm{irr}$",
        default_fit=False,
        default_bounds=[1300, 2500],
    )
    def equilTemperature(self) -> float:  # noqa: N802
        """Planet equilibrium temperature"""
        return self.T_irr

    @equilTemperature.setter
    def equilTemperature(self, value: float) -> None:  # noqa: N802
        self.T_irr = value

    @fitparam(
        param_name="kappa_irr",
        param_latex=r"$k_\\mathrm{irr}$",
        default_fit=False,
        default_bounds=[1e-10, 1],
        default_mode="log",
    )
    def meanInfraOpacity(self) -> float:  # noqa: N802
        """mean infra-red opacity"""
        return self.kappa_ir

    @meanInfraOpacity.setter
    def meanInfraOpacity(self, value: float) -> None:  # noqa: N802
        self.kappa_ir = value

    @fitparam(
        param_name="kappa_v1",
        param_latex=r"$k_\\mathrm{1}$",
        default_fit=False,
        default_bounds=[1e-10, 1],
        default_mode="log",
    )
    def meanOpticalOpacity1(self) -> float:  # noqa: N802
        """mean optical opacity one"""
        return self.kappa_v1

    @meanOpticalOpacity1.setter
    def meanOpticalOpacity1(self, value: float) -> None:  # noqa: N802
        """mean optical opacity one"""
        self.kappa_v1 = value

    @fitparam(
        param_name="kappa_v2",
        param_latex=r"$k_\\mathrm{2}$",
        default_fit=False,
        default_bounds=[1e-10, 1],
        default_mode="log",
    )
    def meanOpticalOpacity2(self) -> float:  # noqa: N802
        """mean optical opacity two"""
        return self.kappa_v2

    @meanOpticalOpacity2.setter
    def meanOpticalOpacity2(self, value: float) -> None:  # noqa: N802
        """mean optical opacity two"""
        self.kappa_v2 = value

    @fitparam(
        param_name="alpha",
        param_latex=r"$\\alpha$",
        default_fit=False,
        default_bounds=[0.0, 1.0],
    )
    def opticalRatio(self) -> float:  # noqa: N802
        """ratio between kappa_v1 and kappa_v2."""
        return self.alpha

    @opticalRatio.setter
    def opticalRatio(self, value: float) -> None:  # noqa: N802
        self.alpha = value

    @fitparam(
        param_name="T_int_guillot",
        param_latex="$T^{g}_{int}$",
        default_fit=False,
        default_bounds=[0.0, 1.0],
    )
    def internalTemperature(self) -> float:  # noqa: N802
        """ratio between kappa_v1 and kappa_v2"""
        return self.T_int

    @internalTemperature.setter
    def internalTemperature(self, value: float) -> None:  # noqa: N802
        """ratio between kappa_v1 and kappa_v2"""
        self.T_int = value

    def _check_values(self) -> None:
        """Ensures kappa values are valid.

        Raises
        ------
        InvalidModelException:
            If any kappa is zero

        """
        if self.kappa_ir == 0.0:
            self.warning("Kappa ir is zero")
            raise InvalidModelException("kappa_ir is zero")

        gamma_1 = self.kappa_v1 / (self.kappa_ir)
        gamma_2 = self.kappa_v2 / (self.kappa_ir)

        if gamma_1 == 0.0 or gamma_2 == 0.0:
            self.warning(
                "Gamma is zero. kappa_v1 = %s kappa_v2 = %s" " kappa_ir = %s",
                self.kappa_v1,
                self.kappa_v2,
                self.kappa_ir,
            )
            raise InvalidModelException("Kappa v1/v2/ir values result in zero gamma")

        if self.T_irr < 0 or self.T_int < 0:
            self.warning(
                "Negative temperature input T_irr=%s T_int=%s", self.T_irr, self.T_int
            )
            raise InvalidModelException("Negative temperature input")

    @property
    def profile(self) -> npt.NDArray[np.float64]:
        """Returns a guillot temperature temperature profile.

        Returns
        --------
        temperature_profile :
            Temperature profile at each layer in Kelvin.

        """

        planet_grav = self.planet.gravity

        self._check_values()

        gamma_1 = self.kappa_v1 / (self.kappa_ir)
        gamma_2 = self.kappa_v2 / (self.kappa_ir)
        tau = self.kappa_ir * self.pressure_profile / planet_grav

        t_int = self.T_int  # todo internal heat parameter looks suspicious..

        def eta(gamma, tau):
            import scipy.special as spe

            part1 = 2.0 / 3.0 + 2.0 / (3.0 * gamma) * (
                1.0 + (gamma * tau / 2.0 - 1.0) * np.exp(-1.0 * gamma * tau)
            )

            part2 = (
                2.0 * gamma / 3.0 * (1.0 - tau**2 / 2.0) * spe.expn(2, (gamma * tau))
            )

            return part1 + part2

        t4 = (
            3.0 * t_int**4 / 4.0 * (2.0 / 3.0 + tau)
            + 3.0 * self.T_irr**4 / 4.0 * (1.0 - self.alpha) * eta(gamma_1, tau)
            + 3.0 * self.T_irr**4 / 4.0 * self.alpha * eta(gamma_2, tau)
        )

        return t4**0.25

    def write(self, output: OutputGroup) -> OutputGroup:
        """Write temperature profile to output."""
        temperature = super().write(output)
        temperature.write_scalar("T_irr", self.T_irr)
        temperature.write_scalar("kappa_irr", self.kappa_ir)
        temperature.write_scalar("kappa_v1", self.kappa_v1)
        temperature.write_scalar("kappa_v2", self.kappa_v2)
        temperature.write_scalar("alpha", self.alpha)
        return temperature

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Return all input keywords."""
        return (
            "guillot",
            "guillot2010",
        )

    BIBTEX_ENTRIES = [
        r"""
        @article{guillot,
        author = {{Guillot, T.}},
        title = {On the radiative equilibrium of irradiated planetary atmospheres},
        DOI= "10.1051/0004-6361/200913396",
        url= "https://doi.org/10.1051/0004-6361/200913396",
        journal = {A\&A},
        year = 2010,
        volume = 520,
        pages = "A27",
        month = "",
        }
        """
    ]

"""Opacity for hydrogen ion."""
import typing as t
import warnings

import numpy as np
import numpy.typing as npt

from taurex.exceptions import InvalidModelException
from taurex.model import OneDForwardModel
from taurex.output import OutputGroup

from .contribution import Contribution


class HydrogenIon(Contribution):
    """Class to build opacity for hydrogen ion."""

    def __init__(self) -> None:
        """Constructor."""
        super().__init__("HydrogenIon")
        self.photo_thresh = 1.6419
        self.alpha = 1.438776877e-2 * 1e6

    def f(self, lamb: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Dont remember what this was."""
        c_n = np.array([1, 52.519, 49.534, -118.858, 92.536, -34.194, 4.982])
        n = np.arange(c_n.shape[0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            to_ret = np.sum(
                c_n[..., None]
                * ((1 / lamb[None, ...] - 1 / self.photo_thresh) ** (n[..., None] / 2)),
                axis=0,
            )
        return to_ret
        # return np.nan_to_num(sum([c * ((1 / lamb - 1 / self.photo_thresh) **
        # (n / 2)) for n, c in enumerate(C_n)]))

    def sigma_pd(self, lamb: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Dont remember what this was."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            to_ret = np.nan_to_num(
                1e-18
                * (lamb**3)
                * (1 / lamb - 1 / self.photo_thresh) ** (3 / 2)
                * self._f_res
            )
        return to_ret

    def k_bf(
        self, lamb: npt.NDArray[np.float64], temperature: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute bound-free."""
        return (
            0.750
            * temperature ** (-5 / 2)
            * np.exp(self.alpha / (self.photo_thresh * temperature))
            * (1 - np.exp(-self.alpha / (lamb * temperature)))
            * self.sigma_pd(lamb)
        )

    def precalc_k_ff(self, lamb: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Precalculate free-free (For performance)."""
        c_3a = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [2483.3460, 285.8270, -2054.2910, 2827.7760, -1341.5370, 208.9520],
                [-3449.8890, -1158.3820, 8746.5230, -11485.6320, 5303.6090, -812.9390],
                [2200.0400, 2427.7190, -13651.1050, 16755.5240, -7510.4940, 1132.7380],
                [-696.2710, -1841.4000, 8642.9700, -10051.5300, 4400.0670, -655.0200],
                [88.2830, 444.5170, -1863.8640, 2095.2880, -901.7880, 132.9850],
            ]
        )

        c_3b = np.array(
            [
                [518.1021, -734.8666, 1021.1775, -479.0721, 93.1373, -6.4285],
                [473.2636, 1443.4137, -1977.3395, 922.3575, -178.9275, 12.3600],
                [-482.2089, -737.1616, 1096.8827, -521.1341, 101.7963, -7.0571],
                [115.5291, 169.6374, -245.6490, 114.2430, -21.9972, 1.5097],
            ]
        )

        self.threea_indices = np.where(lamb > 0.3645)[0]
        self.threeb_indices = np.where((0.1823 < lamb) & (lamb <= 0.3645))[0]

        self.ka, self.pa = self.pre_k_ff_coeff(lamb[self.threea_indices], c_3a)
        self.kb, self.pb = self.pre_k_ff_coeff(lamb[self.threeb_indices], c_3b)

    def k_ff(
        self, lamb: npt.NDArray[np.float64], temperature: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute free-free."""
        res = np.zeros_like(lamb)
        res[self.threea_indices] = (
            np.sum(((5040 / temperature) ** self.pa) * self.ka, axis=0) * 1e-29
        )
        res[self.threeb_indices] = (
            np.sum(((5040 / temperature) ** self.pb) * self.kb, axis=0) * 1e-29
        )
        return res

    def pre_k_ff_coeff(
        self, lamb: npt.NDArray[np.float64], coeff: npt.NDArray[np.float64]
    ) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Get coefficients for k for free-free absorption."""
        # k = np.zeros_like(lamb)
        a_coeff = coeff[..., 0]
        b_coeff = coeff[..., 1]
        c_coeff = coeff[..., 2]
        d_coeff = coeff[..., 3]
        e_coeff = coeff[..., 4]
        f_coeff = coeff[..., 5]

        n = np.arange(coeff.shape[0])

        lamb_2 = lamb[None, ...]

        k = (
            (lamb_2**2) * a_coeff[..., None]
            + b_coeff[..., None]
            + c_coeff[..., None] / lamb_2
            + d_coeff[..., None] / (lamb_2**2)
            + e_coeff[..., None] / (lamb_2**3)
            + f_coeff[..., None] / (lamb_2**4)
        )
        ppart = (n[..., None] + 2) / 2
        return k, ppart
        # return 1e-29 * k

    def k_ff_coeff(
        self,
        lamb: npt.NDArray[np.float64],
        temperature: npt.NDArray[np.float64],
        coeff: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Get coefficients for k for free-free absorption."""
        # k = np.zeros_like(lamb)
        a_coeff = coeff[..., 0]
        b_coeff = coeff[..., 1]
        c_coeff = coeff[..., 2]
        d_coeff = coeff[..., 3]
        e_coeff = coeff[..., 4]
        f_coeff = coeff[..., 5]

        n = np.arange(coeff.shape[0])

        lamb_2 = lamb[None, ...]

        k = np.sum(
            ((5040 / temperature) ** ((n[..., None] + 2) / 2))
            * (
                (lamb_2**2) * a_coeff[..., None]
                + b_coeff[..., None]
                + c_coeff[..., None] / lamb_2
                + d_coeff[..., None] / (lamb_2**2)
                + e_coeff[..., None] / (lamb_2**3)
                + f_coeff[..., None] / (lamb_2**4)
            ),
            axis=0,
        )

        return 1e-29 * k

    def prepare_each(
        self, model: OneDForwardModel, wngrid
    ) -> t.Generator[t.Tuple[str, npt.NDArray[np.float64]], None, None]:
        """Computes and weighs cross-section for a H-"""
        self._nlayers = model.nLayers
        self._P_dyne = model.pressureProfile * 1e6

        # Since our pressure is in Pa already:
        self._P_dyne = model.pressureProfile * 1e1
        self._temperature_profile = model.temperatureProfile

        if "H" not in model.chemistry.activeGases + model.chemistry.inactiveGases:
            raise InvalidModelException("No hydrogen in chemical model")

        if "e-" not in model.chemistry.activeGases + model.chemistry.inactiveGases:
            raise InvalidModelException("No free electrons in chemical model")

        # Taking the H and e- profiles from chemistry class.
        # This allows flexibility in the profiles ;)
        self._hydrogen_mixratio = model.chemistry.get_gas_mix_profile("H")
        self.debug("hydrogen %s", self._hydrogen_mixratio)
        self._electron_mixratio = model.chemistry.get_gas_mix_profile("e-")
        self.debug("e- %s", self._electron_mixratio)

        xsec_ff = []
        xsec_bf = []
        self.sigma_xsec = np.zeros(shape=(self._nlayers, wngrid.shape[0]))
        self._f_res = self.f(10000 / wngrid)
        self.precalc_k_ff(10000 / wngrid)
        for i in range(self._nlayers):
            xsec_ff = (
                self.k_ff(10000 / wngrid, self._temperature_profile[i])
                * self._P_dyne[i]
                * self._hydrogen_mixratio[i]
                * self._electron_mixratio[i]
            )
            xsec_bf = (
                self.k_bf(10000 / wngrid, self._temperature_profile[i])
                * self._P_dyne[i]
                * self._hydrogen_mixratio[i]
                * self._electron_mixratio[i]
            )

            self.sigma_xsec[i, :] = xsec_ff[:] + xsec_bf[:]

        # convert cm2 to m2 ??
        self.sigma_xsec *= 1e-4

        self.debug("final xsec %s", self.sigma_xsec)
        self.debug("final xsec %s", self.sigma_xsec.max())

        # self._total_contrib[...]=0.0
        yield "HydrogenIon", self.sigma_xsec

    @property
    def sigma(self) -> npt.NDArray[np.float64]:
        """Cross-section."""
        return self.sigma_xsec

    def write(self, output: OutputGroup) -> OutputGroup:
        """Write values to file."""
        contrib = super().write(output)
        contrib.write_scalar("hydrogen", self._hydrogen_mixratio)
        contrib.write_scalar("e-", self._electron_mixratio)
        return contrib

    @classmethod
    def input_keywords(cls) -> t.Tuple[str]:
        """Return input keywords for the contribution."""
        return ("HydrogenIon",)

    BIBTEX_ENTRIES = [
        """
        @ARTICLE{John88,
       author = {{John}, T.~L.},
                title = "{Continuous absorption by the
                            negative hydrogen ion reconsidered}",
            journal = {\aap},
            keywords = {Absorption Spectra, Astronomical Spectroscopy, Hydrogen Ions,
            Negative Ions, Photodetachment, Absorptivity,
            Electron Transitions, Wave Functions, Astrophysics},
                year = 1988,
                month = mar,
            volume = {193},
            number = {1-2},
                pages = {189-192},
            adsurl = {https://ui.adsabs.harvard.edu/abs/1988A&A...193..189J},
            adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }
        """
    ]

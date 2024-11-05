"""Rodgers 2000 temperature profile."""
import typing as t

import numpy as np
import numpy.typing as npt

from taurex.data.fittable import fitparam
from taurex.output import OutputGroup

from .tprofile import TemperatureProfile


class Rodgers2000(TemperatureProfile):
    """Layer-by-layer temperature introduced in Rodgers et al (2000)

    Inverse Methods for Atmospheric Sounding (equation 3.26).
    Featured in NEMESIS code (Irwin et al., 2008,
    J. Quant. Spec., 109, 1136 (equation 19)
    Used in all Barstow et al. papers.
    """

    def __init__(
        self,
        temperature_layers: t.Optional[npt.ArrayLike] = None,
        correlation_length: t.Optional[float] = 5.0,
        covariance_matrix: t.Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        """Initialize Rodgers 2000 temperature profile.


        Parameters
        ----------
        temperature_layers : :obj:`list`
            Temperature in Kelvin per layer of pressure

        correlation_length : float
            In scaleheights, Line et al. 2013 sets this to 7, Irwin et al sets
            this to 1.5 may be left as free and Pressure dependent parameter later.

        covariance_matrix : :obj:`array` , optional
            User can supply their own covaraince matrix

        """
        super().__init__("Rodgers2000")

        self._tp_corr_length = correlation_length
        self._covariance = covariance_matrix
        self.temperature_layers = np.array(temperature_layers)
        self.generate_temperature_fitting_params()

    def gen_covariance(self) -> npt.NDArray[np.float64]:
        """Generate the covariance matrix if None is supplied."""
        h = self._tp_corr_length
        pres_prof = self.pressure_profile

        return np.exp(
            -1.0 * np.abs(np.log(pres_prof[:, None] / pres_prof[None, :])) / h
        )

    def correlate_temp(self, cov_mat: npt.NDArray[np.float64]) -> np.float64:
        """Correlate the temperature profile using the covariance matrix."""
        cov_mat_sum = np.sum(cov_mat, axis=0)
        weights = cov_mat[:, :] / cov_mat_sum[:, None]
        return weights.dot(self.temperature_layers)

    @property
    def profile(self) -> npt.NDArray[np.float64]:
        """Returns a temperature profile."""
        cov_mat = self._covariance
        if cov_mat is None:
            cov_mat = self.gen_covariance()

        return self.correlate_temp(cov_mat)

    @fitparam(
        param_name="correlation_length",
        param_latex="$C_{L}$",
        default_fit=False,
        default_bounds=[1.0, 10.0],
    )
    def correlationLength(self) -> float:  # noqa: N802
        """Correlation length in scale heights."""
        return self._tp_corr_length

    @correlationLength.setter
    def correlationLength(self, value: float) -> None:  # noqa: N802
        """Correlation length in scale heights."""
        self._tp_corr_length = value

    def generate_temperature_fitting_params(self) -> None:
        """Generates the temperature fitting parameters

        Parameters are generated for each layer of the
        atmosphere For a 4 layer atmosphere the fitting parameters generated
        are ``T_0``, ``T_1``, ``T_2`` and ``T_3``
        """

        bounds = [1e5, 1e3]
        for idx, _ in enumerate(self.temperature_layers):
            point_num = idx + 1
            param_name = f"T_{point_num}"
            param_latex = "$T_{%i}$" % point_num

            def read_point(self, idx=idx):
                return self.temperature_layers[idx]

            def write_point(self, value, idx=idx):
                self.temperature_layers[idx] = value

            read_point.__doc__ = f"""Temperature at layer {point_num} in Kelvin"""

            fget_point = read_point

            fset_point = write_point
            default_fit = False
            self.add_fittable_param(
                param_name,
                param_latex,
                fget_point,
                fset_point,
                "linear",
                default_fit,
                bounds,
            )

    def write(self, output: OutputGroup) -> OutputGroup:
        """Write Rodgers 2000 temperature profile to output group."""
        temperature = super().write(output)

        cov_mat = self._covariance
        if cov_mat is None:
            cov_mat = self.gen_covariance()

        temperature.write_array("covariance_matrix", cov_mat)
        temperature.write_array("temperature_layers", self.temperature_layers)
        temperature.write_scalar("correlation_length", self._tp_corr_length)
        return temperature

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Return all input keywords."""
        return (
            "rodgers",
            "rodgers2010",
        )

    BIBTEX_ENTRIES = [
        """
        @MISC{rodger_retrievals,
            author = {{Rodgers}, Clive D.},
                title = "{Inverse Methods for Atmospheric
                Sounding - Theory and Practice}",
        howpublished = {Inverse Methods for Atmospheric Sounding - Theory
        and Practice. Series: Series on Atmospheric Oceanic and Planetary Physics},
                year = "2000",
                month = "Jan",
                doi = {10.1142/9789812813718},
            adsurl = {https://ui.adsabs.harvard.edu/abs/2000SAOPP...2.....R},
            adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }
        """,
    ]

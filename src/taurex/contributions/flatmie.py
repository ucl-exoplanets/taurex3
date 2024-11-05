"""Module for computing flat Mie opacity."""
import typing as t

import numpy as np
import numpy.typing as npt

from taurex.data.fittable import fitparam
from taurex.model import OneDForwardModel
from taurex.output import OutputGroup

from .contribution import Contribution


class FlatMieContribution(Contribution):
    """Computes a flat (gray) absorption contribution.

    Absorption is computed as a flat value between two pressures
    across all wavenumbers.

    Parameters
    ----------

    flat_mix_ratio: float
        Opacity value

    flat_bottomP: float
        Bottom of absorbing region in Pa

    flat_topP: float
        Top of absorbing region in Pa

    """

    def __init__(
        self,
        flat_mix_ratio: float = 1e-10,
        flat_bottomP: float = -1,  # noqa: N803
        flat_topP: float = -1,
    ) -> None:
        super().__init__("Mie")

        self._mie_mix = flat_mix_ratio
        self._mie_bottom_pressure = flat_bottomP
        self._mie_top_pressure = flat_topP

    @fitparam(
        param_name="flat_topP",
        param_latex=r"$P^{mie}_\mathrm{top}$",
        default_mode="log",
        default_fit=False,
        default_bounds=[1e-20, 1],
    )
    def mieTopPressure(self) -> float:  # noqa: N802
        """
        Pressure at top of absorbing region in Pa
        """
        return self._mie_top_pressure

    @mieTopPressure.setter
    def mieTopPressure(self, value: float) -> None:  # noqa: N802
        self._mie_top_pressure = value

    @fitparam(
        param_name="flat_bottomP",
        param_latex=r"$P^{mie}_\mathrm{bottom}$",
        default_mode="log",
        default_fit=False,
        default_bounds=[1e-20, 1],
    )
    def mieBottomPressure(self) -> float:  # noqa: N802
        """Pressure at bottom of absorbing region in Pa."""
        return self._mie_bottom_pressure

    @mieBottomPressure.setter
    def mieBottomPressure(self, value: float) -> None:  # noqa: N802
        self._mie_bottom_pressure = value

    @fitparam(
        param_name="flat_mix_ratio",
        param_latex=r"$\chi_\mathrm{mie}$",
        default_mode="log",
        default_fit=False,
        default_bounds=[1e-20, 1],
    )
    def mieMixing(self) -> float:  # noqa: N802
        """Opacity of absorbing region in :math:`m^2`."""
        return self._mie_mix

    @mieMixing.setter
    def mieMixing(self, value: float) -> None:  # noqa: N802
        self._mie_mix = value

    def prepare_each(
        self, model: OneDForwardModel, wngrid: npt.NDArray[np.float64]
    ) -> t.Generator[t.Tuple[str, npt.NDArray[np.float64]], None, None]:
        """Computes and flat absorbing opacity for the pressure regions given.


        Parameters
        ----------
        model: :class:`~taurex.model.model.ForwardModel`
            Forward model

        wngrid: :obj:`array`
            Wavenumber grid

        Yields
        ------
        component: :obj:`tuple` of type (str, :obj:`array`)
            ``Flat`` and the weighted mie opacity.


        """
        self._nlayers = model.nLayers
        self._ngrid = wngrid.shape[0]

        pressure_levels = np.log10(model.pressure.pressure_profile_levels[::-1])

        bottom_pressure = self.mieBottomPressure
        if bottom_pressure < 0:
            bottom_pressure = pressure_levels.max()

        top_pressure = np.log10(self.mieTopPressure)
        if top_pressure < 0:
            top_pressure = pressure_levels.min()

        p_left = pressure_levels[:-1]
        p_right = pressure_levels[1:]

        p_range = sorted([top_pressure, bottom_pressure])

        save_start = np.searchsorted(p_right, p_range[0], side="right")
        save_stop = np.searchsorted(p_left[1:], p_range[1], side="right")
        p_min = p_left[save_start : save_stop + 1]
        p_max = p_right[save_start : save_stop + 1]
        weight = np.minimum(p_range[-1], p_max) - np.maximum(p_range[0], p_min)
        weight /= weight.max()
        sigma_xsec = np.zeros(shape=(self._nlayers, wngrid.shape[0]))
        sigma_xsec[save_start : save_stop + 1] = weight[:, None] * self.mieMixing

        sigma_xsec = sigma_xsec[::-1]

        self.sigma_xsec = sigma_xsec

        yield "Flat", sigma_xsec

    def write(self, output: OutputGroup) -> OutputGroup:
        """Write contribution to output.

        Parameters
        ----------
        output: :class:`~taurex.output.output.Output`
            Output object to write to

        Returns
        -------
        output: :class:`~taurex.output.output.Output`
            Output object that was written to

        """
        contrib = super().write(output)
        contrib.write_scalar("flat_mix_ratio", self._mie_mix)
        contrib.write_scalar("flat_bottomP", self._mie_bottom_pressure)
        contrib.write_scalar("flat_topP", self._mie_top_pressure)
        return contrib

    @classmethod
    def input_keywords(cls) -> t.Tuple[str]:
        """Return input keywords for the contribution."""
        return ("FlatMie",)

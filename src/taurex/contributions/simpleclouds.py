"""Flat Cloud scattering."""

import typing as t

import numpy as np
import numpy.typing as npt

from taurex.data.fittable import fitparam
from taurex.model import OneDForwardModel
from taurex.output import OutputGroup

from .contribution import Contribution


class SimpleCloudsContribution(Contribution):
    r"""Optically thick cloud deck up to a certain height.

    These have the form:

    .. math::

            \tau(\lambda,z) =
                \begin{cases}
                \infty       & \quad \text{if } P(z) >= P_{0}\\
                0            & \quad \text{if } P(z) < P_{0}
                \end{cases}

    Where :math:`P_{0}` is the pressure at the top of the cloud-deck.


    """

    def __init__(self, clouds_pressure: t.Optional[float] = 1e3) -> None:
        """Initialize the cloud model.

        Parameters
        ----------
        clouds_pressure : float, optional
            Pressure at top of cloud deck, by default 1e3


        """
        super().__init__("SimpleClouds")
        self._cloud_pressure = clouds_pressure

    @property
    def order(self) -> int:
        """Return order of contribution."""
        return 3

    def contribute(
        self,
        model: OneDForwardModel,
        start_layer: int,
        end_layer: int,
        density_offset: int,
        layer: int,
        density: npt.NDArray[np.float64],
        tau: npt.NDArray[np.float64],
        path_length: t.Optional[npt.NDArray[np.float64]] = None,
    ):
        """Contribute the cloud opacity to the optical depth.

        Parameters
        ----------
        model : OneDForwardModel
            The forward model
        start_layer : int
            Initial integration layer
        end_layer : int
            Final integration layer
        density_offset : int
            offset in density layer
        layer : int
            atmospheric layer being computed
        density : npt.NDArray[np.float64]
            density profile of atmosphere
        tau : npt.NDArray[np.float64]
            optical depth to store result
        path_length : npt.NDArray[np.float64], optional
            integration length

        """
        tau[layer] += self.sigma_xsec[layer, :]

    def prepare_each(
        self, model: OneDForwardModel, wngrid: npt.NDArray[np.float64]
    ) -> t.Generator[t.Tuple[str, npt.NDArray[np.float64]], None, None]:
        """Compute cross-section that is infinitely absorbing.

        Absorption occurs up to a certain height.

        Parameters
        ----------
        model: :class:`~taurex.model.model.ForwardModel`
            Forward model

        wngrid: :obj:`array`
            Wavenumber grid

        Yields
        ------
        component: :obj:`tuple` of type (str, :obj:`array`)
            ``Clouds`` and opacity array.

        """
        contrib = np.zeros(
            shape=(
                model.nLayers,
                wngrid.shape[0],
            )
        )
        cloud_filtr = model.pressureProfile >= self._cloud_pressure
        contrib[cloud_filtr, :] = np.inf
        self._contrib = contrib
        yield "Clouds", self._contrib

    @fitparam(
        param_name="clouds_pressure",
        param_latex=r"$P_\mathrm{clouds}$",
        default_mode="log",
        default_fit=False,
        default_bounds=[1e-3, 1e6],
    )
    def cloudsPressure(self) -> float:  # noqa: N802
        """Cloud top pressure in Pascal."""
        return self._cloud_pressure

    @cloudsPressure.setter
    def cloudsPressure(self, value: float) -> None:  # noqa: N802
        """Cloud top pressure in Pascal."""
        self._cloud_pressure = value

    def write(self, output: OutputGroup) -> OutputGroup:
        """Write the cloud pressure to the output.

        Parameters
        ----------
        output : :class:`~taurex.output.output.OutputGroup`
            Output group to write to.

        Returns
        -------
        :class:`~taurex.output.output.OutputGroup`
            Output group written to.

        """
        contrib = super().write(output)
        contrib.write_scalar("clouds_pressure", self._cloud_pressure)
        return contrib

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, str]:
        """Return input keywords for the contribution."""
        return (
            "SimpleClouds",
            "ThickClouds",
        )

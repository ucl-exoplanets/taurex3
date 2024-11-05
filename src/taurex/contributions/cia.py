"""Opacity integration of collision-induced absorption."""
import typing as t

import numpy as np
import numpy.typing as npt

from taurex.cache import CIACache
from taurex.model import OneDForwardModel
from taurex.output import OutputGroup

from .contribution import Contribution

contribute_cia: t.Callable[
    [
        int,
        int,
        int,
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        int,
        int,
        int,
        npt.NDArray[np.float64],
    ],
    npt.NDArray[np.float64],
] = None


def contribute_cia_numpy(
    startk: int,
    endk: int,
    density_offset: int,
    sigma: npt.NDArray[np.float64],
    density: npt.NDArray[np.float64],
    path: npt.NDArray[np.float64],
    nlayers: int,
    ngrid: int,
    layer: int,
    tau: npt.NDArray[np.float64],
):
    """Integrate the optical depth for a given layer."""
    _path = path[startk:endk, None]
    _density = density[startk + density_offset : endk + density_offset, None]
    _sigma = sigma[startk + layer : endk + layer, :]

    tau[layer, :] += np.sum(_sigma * _path * _density * _density, axis=0)


def contribute_cia_numba(
    startk: int,
    endk: int,
    density_offset: int,
    sigma: npt.NDArray[np.float64],
    density: npt.NDArray[np.float64],
    path: npt.NDArray[np.float64],
    nlayers: int,
    ngrid: int,
    layer: int,
    tau: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Collisionally induced absorption integration function.

    This has the form:

    .. math::

        \\tau_{\\lambda}(z) = \\int_{z_{0}}^{z_{1}}
            \\sigma(z') \\rho(z')^{2} dz',

    where :math:`z` is the layer, :math:`z_0` and :math:`z_1` are ``startK``
    and ``endK`` respectively. :math:`\\sigma` is the weighted
    cross-section ``sigma``. :math:`rho` is the ``density`` and
    :math:`dz'` is the integration path length ``path``


    Parameters
    ----------
    startK: int
        starting layer in integration

    endK: int
        last layer in integration

    density_offset: int
        Which part of the density profile to start from

    sigma: :obj:`array`
        cross-section

    density: array_like
        density profile of atmosphere

    path: array_like
        path-length or altitude gradient

    nlayers: int
        Total number of layers (unused)

    ngrid: int
        total number of grid points

    layer: int
        Which layer we currently on

    Returns
    -------
    tau : array_like
        optical depth (well almost you still need to do ``exp(-tau)`` yourself)

    """
    for k in range(startk, endk):
        _path = path[k]
        _density = density[k + density_offset]
        # for mol in range(nmols):
        for wn in range(ngrid):
            tau[layer, wn] += sigma[k + layer, wn] * _path * _density * _density


try:
    import numba

    contribute_cia = numba.jit(contribute_cia_numba, nopython=True, nogil=True)

except ImportError:
    # Non numba version.
    contribute_cia = contribute_cia_numpy


class CIAContribution(Contribution):
    """Computes the CIA contribution to the optical depth.

    CIA is collisionally induced absorption.
    """

    def __init__(self, cia_pairs: t.Optional[t.List[str]] = None) -> None:
        """Initialize CIA.

        Parameters
        ----------
        cia_pairs: :obj:`list` of str
            list of molecule pairs of the form ``mol1-mol2``
            e.g. ``H2-He``

        """
        super().__init__("CIA")
        self._cia_pairs = cia_pairs

        self._cia_cache = CIACache()
        if self._cia_pairs is None:
            self._cia_pairs = []

    @property
    def ciaPairs(self) -> t.Sequence[str]:  # noqa: N802
        """Returns list of molecular pairs involved.

        Returns
        -------
        :obj:`list` of str
            list of molecule pairs of the form ``mol1-mol2``
            e.g. ``H2-He``
        """

        return self._cia_pairs

    @ciaPairs.setter
    def ciaPairs(self, value: t.List[str]) -> None:  # noqa: N806,N802
        """Sets list of molecular pairs involved.

        Parameters
        ----------
        value: :obj:`list` of str
            list of molecule pairs of the form ``mol1-mol2``
            e.g. ``H2-He``

        """
        self._cia_pairs = value

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
    ) -> None:
        """Integrate the optical depth for a given layer."""
        if self._total_cia > 0:
            contribute_cia(
                start_layer,
                end_layer,
                density_offset,
                self.sigma_xsec,
                density,
                path_length,
                self._nlayers,
                self._ngrid,
                layer,
                tau,
            )

    def prepare_each(
        self, model: OneDForwardModel, wngrid: npt.NDArray[np.float64]
    ) -> t.Generator[t.Tuple[str, npt.NDArray[np.float64]], None, None]:
        """Computes and weighs cross-section for a single pair of molecules

        Parameters
        ----------
        model: :class:`~taurex.model.model.ForwardModel`
            Forward model

        wngrid: :obj:`array`
            Wavenumber grid


        Yields
        ------
        component: :obj:`tuple` of type (str, :obj:`array`)
            Molecular pair and the weighted cia opacity.

        """
        self._total_cia = len(self.ciaPairs)
        self._nlayers = model.nLayers
        self._ngrid = wngrid.shape[0]
        self.info("Computing CIA ")

        sigma_cia = np.zeros(shape=(model.nLayers, wngrid.shape[0]))

        chemistry = model.chemistry

        for pair_name in self.ciaPairs:
            cia = self._cia_cache[pair_name]
            sigma_cia[...] = 0.0

            cia_factor = chemistry.get_gas_mix_profile(
                cia.pairOne
            ) * chemistry.get_gas_mix_profile(cia.pairTwo)

            for idx_layer, temperature in enumerate(model.temperatureProfile):
                _cia_xsec = cia.cia(temperature, wngrid)
                sigma_cia[idx_layer] += _cia_xsec * cia_factor[idx_layer]
            self.sigma_xsec = sigma_cia
            yield pair_name, sigma_cia

    def write(self, output: OutputGroup) -> OutputGroup:
        """Write output to file."""
        contrib = super().write(output)
        if len(self.ciaPairs) > 0:
            contrib.write_string_array("cia_pairs", self.ciaPairs)
        return contrib

    @classmethod
    def input_keywords(cls) -> t.Tuple[str]:
        """Return list of input keywords for CIA."""
        return ("CIA",)

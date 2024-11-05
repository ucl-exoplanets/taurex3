"""Handling of molecular absorption."""
import math
import typing as t

import numpy as np
import numpy.typing as npt

from taurex.cache import GlobalCache, OpacityCache
from taurex.cache.ktablecache import KTableCache
from taurex.model.model import ForwardModel

from .contribution import Contribution

contribute_ktau: t.Callable[
    [
        int,
        int,
        int,
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        int,
        int,
        int,
    ],
    npt.NDArray[np.float64],
] = None


def contribute_ktau_numba(
    startk: int,
    endk: int,
    density_offset: int,
    sigma: npt.NDArray[np.float64],
    density: npt.NDArray[np.float64],
    path: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    tau: npt.NDArray[np.float64],
    ngrid: int,
    layer: int,
    ngauss: int,
) -> npt.NDArray[np.float64]:
    """Integrate the optical depth for a given layer.

    This version uses numba to speed up the calculation.

    Parameters
    ----------
    startk : int
        Starting integration index of the layer
    endk : int
        Ending integration index of the layer
    density_offset : int
        Offset of the density profile
    sigma : npt.NDArray[np.float64]
        Cross-sections
    density : npt.NDArray[np.float64]
        Density profile
    path : npt.NDArray[np.float64]
        Path length
    weights : npt.NDArray[np.float64]
        Weights for the gauss quadrature
    tau : npt.NDArray[np.float64]
        Optical depth
    ngrid : int
        Number of grid points
    layer : int
        Layer index
    ngauss : int
        Number of gauss points

    """
    tau_temp = np.zeros(shape=(ngrid, ngauss))

    for k in range(startk, endk):
        _path = path[k]
        _density = density[k + density_offset]
        # for mol in range(nmols):
        for wn in range(ngrid):
            for g in range(ngauss):
                tau_temp[wn, g] += sigma[k + layer, wn, g] * _path * _density

    for wn in range(ngrid):
        transtemp = 0.0
        for g in range(ngauss):
            transtemp += math.exp(-tau_temp[wn, g]) * weights[g]
        tau[layer, wn] += -math.log(transtemp)


def contribute_ktau_numpy(
    startk: int,
    endk: int,
    density_offset: int,
    sigma: npt.NDArray[np.float64],
    density: npt.NDArray[np.float64],
    path: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    tau: npt.NDArray[np.float64],
    ngrid: int,  # For compatibility
    layer: int,
    ngauss: int,  # For compatibility
) -> npt.NDArray[np.float64]:
    """Integrate the optical depth for a given layer with numpy.

    This version uses numba to speed up the calculation.

    Parameters
    ----------
    startk : int
        Starting integration index of the layer
    endk : int
        Ending integration index of the layer
    density_offset : int
        Offset of the density profile
    sigma : npt.NDArray[np.float64]
        Cross-sections
    density : npt.NDArray[np.float64]
        Density profile
    path : npt.NDArray[np.float64]
        Path length
    weights : npt.NDArray[np.float64]
        Weights for the gauss quadrature
    tau : npt.NDArray[np.float64]
        Optical depth
    ngrid : int
        Number of grid points
    layer : int
        Layer index
    ngauss : int
        Number of gauss points

    """
    _path = path[startk:endk, None, None]
    _density = density[startk + density_offset : endk + density_offset]
    _sigma = sigma[layer + startk : layer + endk]
    tau_temp = np.sum(
        _sigma * _path[..., None, None] * _density[..., None, None],
        axis=0,
    )

    transtemp = np.sum(np.exp(-tau_temp) * weights[None, :], axis=-1)

    tau[layer] += -np.log(transtemp)

    return tau


try:
    import numba

    contribute_ktau = numba.jit(
        contribute_ktau_numba, nopython=True, nogil=True, fastmath=True
    )
except ImportError:
    contribute_ktau = contribute_ktau_numpy


class AbsorptionContribution(Contribution):
    """Absorption contribution."""

    def contribute(
        self,
        model: ForwardModel,
        start_horz_layer: int,
        end_horz_layer: int,
        density_offset: int,
        layer: int,
        density: npt.NDArray[np.float64],
        tau: npt.NDArray[np.float64],
        path_length: t.Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        """Contribute to the optical depth.

        Parameters
        ----------
        model : ForwardModel
            Forward model
        start_horz_layer : int
            Starting horizontal layer
        end_horz_layer : int
            Ending horizontal layer
        density_offset : int
            Offset of the density profile
        layer : int
            Layer index
        density : npt.NDArray[np.float64]
            Density profile
        tau : npt.NDArray[np.float64]
            Optical depth
        path_length : npt.NDArray[np.float64], optional
            Path length, by default None
        """
        if self._use_ktables:
            # startK,endK,density_offset,sigma,density,path,weights,tau,ngrid,layer,ngauss

            contribute_ktau(
                start_horz_layer,
                end_horz_layer,
                density_offset,
                self.sigma_xsec,
                density,
                path_length,
                self.weights,
                tau,
                self._ngrid,
                layer,
                self.weights.shape[0],
            )
        else:
            super().contribute(
                model,
                start_horz_layer,
                end_horz_layer,
                density_offset,
                layer,
                density,
                tau,
                path_length,
            )

    def __init__(self) -> None:
        super().__init__("Absorption")
        self._opacity_cache = OpacityCache()

    def prepare_each(
        self, model: ForwardModel, wngrid: npt.NDArray[np.float64]
    ) -> t.Generator[t.Tuple[str, npt.NDArray[np.float64]], None, None]:
        """Prepare each component opacity.

        Parameters
        ----------
        model : ForwardModel
            Forward model

        wngrid : npt.NDArray[np.float64]
            Wavenumber grid

        Yields
        ------
        t.Generator[t.Tuple[str, npt.NDArray[np.float64]], None, None]
            Gas name and opacity

        """
        self.debug("Preparing model with %s", wngrid.shape)
        self._ngrid = wngrid.shape[0]
        self._use_ktables = GlobalCache()["opacity_method"] == "ktables"
        self.info("Using cross-sections? %s", not self._use_ktables)

        if self._use_ktables:
            self._opacity_cache = KTableCache()
        else:
            self._opacity_cache = OpacityCache()
        sigma_xsec = None
        self.weights = None

        for gas in model.chemistry.activeGases:
            # self._total_contrib[...] =0.0
            gas_mix = model.chemistry.get_gas_mix_profile(gas)
            self.info("Recomputing active gas %s opacity", gas)

            xsec = self._opacity_cache[gas]

            if self._use_ktables and self.weights is None:
                self.weights = xsec.weights

            if sigma_xsec is None:
                if self._use_ktables:
                    sigma_xsec = np.zeros(
                        shape=(self._nlayers, self._ngrid, len(self.weights))
                    )
                else:
                    sigma_xsec = np.zeros(shape=(self._nlayers, self._ngrid))
            else:
                sigma_xsec[...] = 0.0

            for idx_layer, tp in enumerate(
                zip(model.temperatureProfile, model.pressureProfile)
            ):
                self.debug("Got index,tp %s %s", idx_layer, tp)

                temperature, pressure = tp
                # print(gas,self._opacity_cache[gas].opacity(temperature,pressure,wngrid),gas_mix[idx_layer])
                sigma_xsec[idx_layer] += (
                    xsec.opacity(temperature, pressure, wngrid) * gas_mix[idx_layer]
                )

            self.sigma_xsec = sigma_xsec

            self.debug("SIGMAXSEC %s", self.sigma_xsec)

            yield gas, sigma_xsec

    def prepare(self, model: ForwardModel, wngrid: npt.NDArray[np.float64]) -> None:
        """Used to prepare the contribution for the calculation.
        Called before the forward model performs the main optical depth
        calculation. Default behaviour is to loop through :func:`prepare_each`
        and sum all results into a single cross-section.

        Parameters
        ----------
        model: :class:`~taurex.model.model.ForwardModel`
            Forward model

        wngrid: :obj:`array`
            Wavenumber grid
        """

        self._ngrid = wngrid.shape[0]
        self._nlayers = model.nLayers

        sigma_xsec = None
        self.debug("ABSORPTION VERSION")
        for gas, sigma in self.prepare_each(model, wngrid):
            self.debug("Gas %s", gas)
            self.debug("Sigma %s", sigma)
            if sigma_xsec is None:
                sigma_xsec = np.zeros_like(sigma)
            sigma_xsec += sigma

        self.sigma_xsec = sigma_xsec
        self.debug("Final sigma is %s", self.sigma_xsec)
        self.info("Done")

    def finalize(self, model: ForwardModel) -> None:
        """Finalize the contribution."""
        raise NotImplementedError

    @property
    def sigma(self) -> npt.NDArray[np.float64]:
        """Return effective cross-sections."""
        return self.sigma_xsec

    @classmethod
    def input_keywords(cls) -> t.Tuple[str]:
        """Return input keywords for the contribution."""
        return (
            "Absorption",
            "Molecules",
        )

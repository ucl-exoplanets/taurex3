"""Rayleigh scattering contribution."""
import typing as t

import numpy as np
import numpy.typing as npt

from taurex.model import OneDForwardModel

from .contribution import Contribution


class RayleighContribution(Contribution):
    """Computes contribution from Rayleigh scattering."""

    def __init__(self):
        """Initialise contribution."""
        super().__init__("Rayleigh")

    def prepare_each(
        self, model: OneDForwardModel, wngrid: npt.NDArray[np.float64]
    ) -> t.Generator[t.Tuple[str, npt.NDArray[np.float64]], None, None]:
        """Compute opacity due to rayleigh scattering.

        Scattering is weighted by the mixing ratio of the gas
        from chemistry.

        Parameters
        ----------
        model: :class:`~taurex.model.model.ForwardModel`
            Forward model

        wngrid: :obj:`array`
            Wavenumber grid

        Yields
        ------
        component: :obj:`tuple` of type (str, :obj:`array`)
            Name of scattering molecule and the weighted rayeligh opacity.


        """
        from taurex.util.scattering import rayleigh_sigma_from_name

        self._ngrid = wngrid.shape[0]
        self._nmols = 1
        self._nlayers = model.nLayers
        molecules = list(model.chemistry.activeGases) + list(
            model.chemistry.inactiveGases
        )

        for gasname in molecules:
            if np.max(model.chemistry.get_gas_mix_profile(gasname)) == 0.0:
                continue
            sigma = rayleigh_sigma_from_name(gasname, wngrid)

            if sigma is not None:
                final_sigma = (
                    sigma[None, :]
                    * model.chemistry.get_gas_mix_profile(gasname)[:, None]
                )
                self.sigma_xsec = final_sigma
                yield gasname, final_sigma

    @classmethod
    def input_keywords(cls) -> t.Tuple[str]:
        """Return list of keywords for contribution."""
        return ("Rayleigh",)

    BIBTEX_ENTRIES = [
        """
        @book{cox_allen_rayleigh,
        title={Allen’s astrophysical quantities},
        author={Cox, Arthur N},
        year={2015},
        publisher={Springer}
        }
        """
    ]

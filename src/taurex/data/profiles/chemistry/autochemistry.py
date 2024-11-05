"""Chemistry that automatically seperates out active and inactive gases."""
import typing as t

import numpy as np
import numpy.typing as npt

from .chemistry import Chemistry


class AutoChemistry(Chemistry):
    """
    Chemistry class that automatically seperates out
    active and inactive gases

    Has a helper function that should be called at the
    end of initialization. :func:`determine_active_inactive`
    once :func:`gases` has been set.

    You'll only need to implement :func:`gases` and :func:`mixProfile`
    to use this class.

    """

    def __init__(self, name):
        """Initialize auto chemistry.

        Parameters
        ----------

        name: str
            Name of class

        """
        super().__init__(name)

        self._active = []
        self._inactive = []
        self._inactive_mask = None
        self._active_mask = None

    def determine_active_inactive(self) -> None:
        """Determines active and inactive gases."""
        try:
            self._active, self._active_mask = zip(
                *[(m, i) for i, m in enumerate(self.gases) if m in self.availableActive]
            )
        except ValueError:
            self.debug("No active gases detected")
            self._active, self._active_mask = [], None

        try:
            self._inactive, self._inactive_mask = zip(
                *[
                    (m, i)
                    for i, m in enumerate(self.gases)
                    if m not in self.availableActive
                ]
            )
        except ValueError:
            self.debug("No inactive gases detected")
            self._inactive, self._inactive_mask = [], None

        if self._active_mask is not None:
            self._active_mask = np.array(self._active_mask)

        if self._inactive_mask is not None:
            self._inactive_mask = np.array(self._inactive_mask)

    def compute_mu_profile(self, nlayers: t.Optional[int] = None) -> None:
        """Computes molecular weight of atmosphere for each layer

        Parameters
        ----------
        nlayers: int
            Number of layers, deprecated
        """

        if self.mixProfile is not None:
            mix_profile = self.mixProfile
            self.mu_profile = np.sum(
                [
                    self.get_molecular_mass(gas) * mix
                    for gas, mix in zip(self.gases, mix_profile)
                ],
                axis=0,
            )
        else:
            raise ValueError("Mix profile not set/computed")

    @property
    def gases(self) -> t.List[str]:
        """List of gases in atmosphere."""
        raise NotImplementedError

    @property
    def mixProfile(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Mix profile of gases."""
        raise NotImplementedError

    @property
    def activeGases(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Actively absorbing gases."""
        return self._active

    @property
    def inactiveGases(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Non absorbing gases."""
        return self._inactive

    @property
    def activeGasMixProfile(self) -> t.Optional[npt.NDArray[np.float64]]:  # noqa: N802
        """Active gas layer by layer mix profile

        Returns
        -------
        active_mix_profile : :obj:`array`
            Active gas mix profile with shape ``(nactive, nlayer)``


        """

        if self.mixProfile is None:
            raise ValueError("No mix profile computed.")

        if self._active_mask is None:
            return None

        return self.mixProfile[self._active_mask]

    @property
    def inactiveGasMixProfile(  # noqa: N802
        self,
    ) -> t.Optional[npt.NDArray[np.float64]]:
        """Inactive gas layer by layer mix profile.

        Returns
        -------
        inactive_mix_profile : :obj:`array`

        """
        if self.mixProfile is None:
            raise Exception("No mix profile computed.")
        if self._inactive_mask is None:
            return None
        return self.mixProfile[self._inactive_mask]

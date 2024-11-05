"""Base chemical profile class."""

import typing as t

import numpy as np
import numpy.typing as npt

from taurex.cache import GlobalCache, OpacityCache
from taurex.cache.ktablecache import KTableCache
from taurex.data.citation import Citable
from taurex.data.fittable import Fittable, derivedparam
from taurex.log import Logger
from taurex.output import OutputGroup
from taurex.output.writeable import Writeable
from taurex.planet import Planet
from taurex.stellar import Star


class Chemistry(Fittable, Logger, Writeable, Citable):
    """Skeleton for defining chemistry.
    *Abstract Class*

    Must implement methods:

    - :func:`activeGases`
    - :func:`inactiveGases`
    - :func:`activeGasMixProfile`
    - :func:`inactiveGasMixProfile`

    *Active* are those that are actively
    absorbing in the atmosphere. In technical terms they are molecules
    that have absorption cross-sections. You can see which molecules
    are able to actively absorb by doing:
    You can find out what molecules can actively absorb by doing:

            >>> avail_active_mols = OpacityCache().find_list_of_molecules()

    Active gases are only determined at initialization and cannot be changed
    during runtime. If you want to change the active gases you will need to
    reinitialize the chemistry class.


    """

    def __init__(self, name: str):
        """Initialize chemistry.

        Parameters
        ----------
        name : str
            Name of chemistry for logging.

        """
        Logger.__init__(self, name)
        Fittable.__init__(self)

        self.mu_profile = None

        if GlobalCache()["opacity_method"] == "ktables":
            self._avail_active = KTableCache().find_list_of_molecules()
        else:
            self._avail_active = OpacityCache().find_list_of_molecules()
        # self._avail_active = OpacityCache().find_list_of_molecules()
        deactive_list = GlobalCache()["deactive_molecules"]
        if deactive_list is not None:
            self._avail_active = [
                k for k in self._avail_active if k not in deactive_list
            ]

    def set_star_planet(self, star: Star, planet: Planet):
        """Supplies the star and planet to chemistryfor photochemistry reasons.

        Does nothing by default

        Parameters
        ----------

        star: :class:`~taurex.data.stellar.star.Star`
            A star object

        planet: :class:`~taurex.data.planet.Planet`
            A planet object


        """
        pass

    @property
    def availableActive(self) -> t.List[str]:  # noqa: N802
        """Returns a list of available actively absorbing molecules.

        Returns
        -------
        molecules: :obj:`list`
            Actively absorbing molecules
        """
        return self._avail_active

    @property
    def activeGases(self) -> t.List[str]:  # noqa: N802
        """Actively absorbing gases.

        **Requires implementation**

        Should return a list of molecule names

        Returns
        -------
        active : :obj:`list`
            List of active gases

        """
        raise NotImplementedError

    @property
    def inactiveGases(self) -> t.List[str]:  # noqa: N802
        """Non absorbing gases.
        **Requires implementation**

        Should return a list of molecule names

        Returns
        -------
        inactive : :obj:`list`
            List of inactive gases

        """
        raise NotImplementedError

    def initialize_chemistry(
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
        pass

    @property
    def activeGasMixProfile(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Mix profile of actively absorbing gases.
        **Requires implementation**

        Should return profiles of shape ``(nactivegases,nlayers)``. Active
        refers to gases that are actively absorbing in the atmosphere.
        Another way to put it these are gases where molecular cross-sections
        are used.

        """

        raise NotImplementedError

    @property
    def inactiveGasMixProfile(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Mixing profile of non absorbing gases.

        **Requires implementation**

        Should return profiles of shape ``(ninactivegases,nlayers)``.

        """
        raise NotImplementedError

    @property
    def muProfile(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Molecular weight for each layer of atmosphere in kg

        Returns
        -------
        mix_profile : :obj:`array`

        """
        if self.mu_profile is None:
            self.compute_mu_profile(None)
        return self.mu_profile

    def get_gas_mix_profile(self, gas_name: str) -> npt.NDArray[np.float64]:
        """Returns the mix profile of a particular gas

        Parameters
        ----------
        gas_name : str
            Name of gas

        Returns
        -------
        mixprofile : :obj:`array`
            Mix profile of gas with shape ``(nlayer)``

        """
        if gas_name in self.activeGases:
            idx = self.activeGases.index(gas_name)
            return self.activeGasMixProfile[idx]
        elif gas_name in self.inactiveGases:
            idx = self.inactiveGases.index(gas_name)
            return self.inactiveGasMixProfile[idx]
        else:
            raise KeyError

    def compute_mu_profile(self, nlayers: t.Optional[int] = None) -> None:
        """Computes molecular weight of atmosphere for each layer

        Parameters
        ----------
        nlayers: int
            Number of layers, deprecated
        """

        active = []
        inactive = []

        if self.activeGasMixProfile is not None:
            active = [
                mix * self.get_molecular_mass(gasname)
                for mix, gasname in zip(self.activeGasMixProfile, self.activeGases)
            ]

        if self.inactiveGasMixProfile is not None:
            inactive = [
                mix * self.get_molecular_mass(gasname)
                for mix, gasname in zip(self.inactiveGasMixProfile, self.inactiveGases)
            ]

        total = active + inactive
        if not total:
            raise ValueError("No gases or chemical profile in atmosphere")

        self.mu_profile = np.sum(total, axis=0)

    @property
    def gases(self) -> t.List[str]:
        """Total list of gases in atmosphere"""
        return self.activeGases + self.inactiveGases

    @property
    def mixProfile(self) -> npt.NDArray[np.float64]:  # noqa: N802
        return np.concatenate((self.activeGasMixProfile, self.inactiveGasMixProfile))

    @derivedparam(param_name="mu", param_latex=r"$\mu$", compute=True)
    def mu(self) -> float:
        """Mean molecular weight at surface (amu)."""
        from taurex.constants import AMU

        return self.muProfile[0] / AMU

    def write(self, output: OutputGroup) -> OutputGroup:
        """Writes chemistry class and arguments to file.

        Parameters
        ----------
        output: :class:`~taurex.output.output.Output`

        """
        gas_entry = output.create_group("Chemistry")
        gas_entry.write_string("chemistry_type", self.__class__.__name__)
        gas_entry.write_string_array("active_gases", self.activeGases)
        gas_entry.write_string_array("inactive_gases", self.inactiveGases)
        if self.hasCondensates:
            gas_entry.write_string_array("condensates", self.condensates)
        return gas_entry

    @property
    def condensates(self) -> t.List[str]:
        """
        Returns a list of condensates in the atmosphere.

        Returns
        -------
        active : :obj:`list`
            List of condensates

        """

        return []

    @property
    def hasCondensates(self) -> bool:  # noqa: N802
        """Returns True if there are condensates in the atmosphere."""
        return len(self.condensates) > 0

    @property
    def condensateMixProfile(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Get condensate mix profile.
        **Requires implementation**

        Should return profiles of shape ``(ncondensates,nlayers)``.
        """
        if len(self.condensates) == 0:
            return None
        else:
            raise NotImplementedError

    def get_condensate_mix_profile(
        self, condensate_name: str
    ) -> npt.NDArray[np.float64]:
        """Returns the mix profile of a particular condensate.

        Parameters
        ----------
        condensate_name : str
            Name of condensate

        Returns
        -------
        mixprofile : :obj:`array`
            Mix profile of condensate with shape ``(nlayer)``

        """
        if condensate_name in self.condensates:
            index = self.condensates.index(condensate_name)
            return self.condensateMixProfile[index]
        else:
            raise KeyError(f"Condensate {condensate_name} not found in chemistry")

    def get_molecular_mass(self, molecule: str) -> float:
        """Returns the molecular mass of a molecule."""
        from taurex.util import get_molecular_weight

        return get_molecular_weight(molecule)

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Input keywords for chemistry class."""
        raise NotImplementedError

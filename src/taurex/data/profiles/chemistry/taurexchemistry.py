"""Main free chemistry model."""
import typing as t

import numpy as np
import numpy.typing as npt

from taurex.core import FittingType, derivedparam
from taurex.exceptions import InvalidModelException
from taurex.output import OutputGroup
from taurex.util import has_duplicates, molecule_texlabel

from .autochemistry import AutoChemistry
from .gas.gas import Gas


class InvalidChemistryException(InvalidModelException):
    """Called when atmosphere mix is greater than unity."""

    pass


class TaurexChemistry(AutoChemistry):
    """The standard chemical model used in Taurex.

    This allows for the combinationof different mixing profiles for each molecule.
    Lets take an example
    profile, we want an atmosphere with a constant mixing of ``H2O`` but two
    layer mixing for ``CH4``.
    First we initialize our chemical model:

        >>> chemistry = TaurexChemistry()

    Then we can add our molecules using the :func:`addGas` method. Lets start
    with ``H2O``, since its a constant profile for all layers of the atmosphere
    we thus add
    the :class:`~taurex.data.profiles.chemistry.gas.constantgas.ConstantGas`
    object:

        >>> chemistry.addGas(ConstantGas('H2O',mix_ratio = 1e-4))

    Easy right? Now the same goes for ``CH4``, we can add the molecule into
    the chemical model by using the correct profile (in this case
    :class:`~taurex.data.profiles.chemistry.gas.twolayergas.TwoLayerGas`):

        >>> chemistry.addGas(TwoLayerGas('CH4',mix_ratio_surface=1e-4,
                                         mix_ratio_top=1e-8))

    Chaining is also supported:

        >>> chemistry.addGas(gas1).addGas(gas2)

    Molecular profiles available are:
        * :class:`~taurex.data.profiles.chemistry.gas.constantgas.ConstantGas`
        * :class:`~taurex.data.profiles.chemistry.gas.twolayergas.TwoLayerGas`
        * :class:`~taurex.data.profiles.chemistry.gas.twolayergas.TwoPointGas`


    """

    def __init__(
        self,
        fill_gases: t.Optional[t.List[str]] = None,
        ratio: t.Union[float, t.List[float]] = 0.17567,
        derived_ratios: t.List[str] = None,
        base_metallicity: float = 0.013,
    ) -> None:
        """Initialize free chemistry.

        Parameters
        ----------

        fill_gases : str or :obj:`list`
            Either a single gas or list of gases to fill the atmosphere with
            default: ``['H2','He']``

        ratio : float or :obj:`list`
            If a bunch of molecules are used to fill an atmosphere, whats the
            ratio between them?
            The first fill gas is considered the main one with others defined as
            ``molecule / main_molecule``
            default: ``0.17567``

        derived_ratios : :obj:`list`
            List of element ratios to compute as derived parameters
            (e.g. ``C/O``)

        base_metallicity : float
            What to consider the base metallicity of the atmosphere
            (default: 0.013)

        """
        super().__init__("ChemistryModel")

        self._gases: t.List[Gas] = []
        self._active = []
        self._inactive = []
        fill_gases = fill_gases or ["H2", "He"]
        derived_ratios = derived_ratios or []
        if isinstance(fill_gases, str):
            fill_gases = [fill_gases]

        if isinstance(ratio, float):
            ratio = [ratio]

        if has_duplicates(fill_gases):
            self.error("Fill gasses has duplicate molecules")
            self.error("Fill gasses: %s", fill_gases)
            raise ValueError("Duplicate fill gases detected")

        if len(fill_gases) > 1 and len(ratio) != len(fill_gases) - 1:
            self.error("Fill gases and ratio count are not correctly matched")
            self.error(
                "There should be %s ratios, you have defined %s",
                len(fill_gases) - 1,
                len(ratio),
            )
            raise InvalidChemistryException

        self._fill_gases = fill_gases
        self._fill_ratio = ratio
        self._mix_profile = None
        self._base_metallicity = base_metallicity
        self.debug("MOLECULES I HAVE %s", self.availableActive)
        self.setup_fill_params()
        self.determine_active_inactive()
        self.setup_derived_params(derived_ratios)

    # def determine_mix_mask(self):

    #     try:
    #         self._active, self._active_mask = zip(*[(m, i) for i, m in
    #                                                     enumerate(self.gases)
    #                                                     if m in
    #                                                     self.availableActive])
    #     except ValueError:
    #         self.debug('No active gases detected')
    #         self._active, self._active_mask = [], None

    #     try:
    #         self._inactive, self._inactive_mask = zip(*[(m, i) for i, m in
    #                                                         enumerate(self.gases)
    #                                                         if m not in
    #                                                         self.availableActive])
    #     except ValueError:
    #         self.debug('No inactive gases detected')
    #         self._inactive, self._inactive_mask = [], None

    #     self._active_mask = np.array(self._active_mask)
    #     self._inactive_mask = np.array(self._inactive_mask)

    def setup_fill_params(self) -> None:
        """Generate fill gas parameters."""
        if not hasattr(self._fill_gases, "__len__") or len(self._fill_gases) < 2:
            return

        main_gas = self._fill_gases[0]

        for idx, value in enumerate(zip(self._fill_gases[1:], self._fill_ratio)):
            gas, ratio = value
            mol_name = f"{gas}_{main_gas}"
            param_name = mol_name
            param_tex = "{}/{}".format(
                molecule_texlabel(gas), molecule_texlabel(main_gas)
            )

            def read_mol(self, idx=idx):
                return self._fill_ratio[idx]

            def write_mol(self, value, idx=idx):
                self._fill_ratio[idx] = value

            fget = read_mol
            fset = write_mol

            fget.__doc__ = f"{gas}/{main_gas} ratio (volume)"

            bounds = [1.0e-12, 0.1]

            default_fit = False
            self.add_fittable_param(
                param_name, param_tex, fget, fset, "log", default_fit, bounds
            )

    def setup_derived_params(self, ratio_list: t.List[str]) -> None:
        """Generate derived parameters."""
        for elem_ratio in ratio_list:
            elem1, elem2 = elem_ratio.split("/")
            mol_name = f"{elem1}_{elem2}_ratio"
            param_name = mol_name
            param_tex = "{}/{}".format(
                molecule_texlabel(elem1), molecule_texlabel(elem2)
            )

            def read_mol(self, elem=elem_ratio):
                return np.mean(self.get_element_ratio(elem))

            fget = read_mol

            fget.__doc__ = f"{elem_ratio} ratio (volume)"

            compute = True
            self.add_derived_param(param_name, param_tex, fget, compute)

    # def compute_mu_profile(self, nlayers):
    #     """
    #     Computes molecular weight of atmosphere
    #     for each layer

    #     Parameters
    #     ----------
    #     nlayers: int
    #         Number of layers
    #     """
    #     from taurex.util import get_molecular_weight
    #     self.mu_profile = np.zeros(shape=(nlayers,))
    #     if self.mixProfile is not None:
    #         mix_profile = self.mixProfile
    #         for idx, gasname in enumerate(self.gases):
    #             self.mu_profile += mix_profile[idx] * \
    #                 get_molecular_weight(gasname)

    def isActive(self, gas: str) -> bool:  # noqa: N802
        """Determines if the gas is absorbing.

        Parameters
        ----------

        gas: str
            Name of molecule


        Returns
        -------
        bool:
            True if active
        """
        if gas in self.availableActive:
            return True
        else:
            return False

    def addGas(self, gas: Gas) -> "TaurexChemistry":  # noqa: N802
        """Adds a gas in the atmosphere.

        Parameters
        ----------
        gas : :class:`~taurex.data.profiles.chemistry.gas.gas.Gas`
            Gas to add into the atmosphere. Only takes effect
            on next initialization call.

        """

        if gas.molecule in [x.molecule for x in self._gases]:
            self.error("Gas already exists %s", gas.molecule)
            raise ValueError("Gas already exists")

        self.debug("Gas %s fill gas: %s", gas.molecule, self._fill_gases)
        if gas.molecule in self._fill_gases:
            self.error(
                "Gas %s is already a fill gas: %s", gas.molecule, self._fill_gases
            )
            raise ValueError("Gas already exists")

        self._gases.append(gas)

        self.determine_active_inactive()

        return self

    @property
    def gases(self) -> t.List[str]:
        """List of gases in the atmosphere."""
        return self._fill_gases + [g.molecule for g in self._gases]

    @property
    def mixProfile(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Mixing profile of all gases."""
        return self._mix_profile

    # @property
    # def activeGases(self):
    #     return self._active

    # @property
    # def inactiveGases(self):
    #     return self._inactive

    def compute_elements_mix(self) -> t.Dict[str, npt.NDArray[np.float64]]:
        """Determines the elemental mix of the atmosphere."""
        from taurex.util import split_molecule_elements

        element_dict = {}

        for g, m in zip(self.gases, self.mixProfile):
            avg_mix = m
            s = [], []
            if g != "e-":
                s = split_molecule_elements(g)
            else:
                s = {"e-": 1}
            # total_count = sum(s.values())

            for elements, count in s.items():
                val = element_dict.get(elements, 0.0)
                element_dict[elements] = val + count * avg_mix

        return element_dict

    @derivedparam(param_name="metallicity", param_latex="Z")
    def metallicity(self) -> float:
        """Metallicity of the atmosphere."""
        return self.get_metallicity()

    def get_metallicity(self) -> float:
        """Metallicity of the atmosphere.

        Determines metallicity of the atmosphere by computing the
        elemental mix of the atmosphere and then determining the
        metallicity by the following formula:

        .. math::

            Z = 1 - M_{H} - M_{He}

        where :math:`M_{H}` is the mass fraction of hydrogen and
        :math:`M_{He}` is the mass fraction of helium.

        Returns
        -------
        float:

        """
        from taurex.constants import AMU
        from taurex.util import mass

        element_dict = self.compute_elements_mix()

        total_mass = self.muProfile.sum() / AMU

        h_mass_fraction = element_dict["H"].sum() * mass["H"] / total_mass

        he_mass_fraction = element_dict["He"].sum() * mass["He"] / total_mass

        metallicity = 1 - h_mass_fraction - he_mass_fraction

        return metallicity / self._base_metallicity

    def get_element_ratio(self, elem_ratio: str) -> npt.NDArray[np.float64]:
        """Get element ratio of atmosphere.

        Parameters
        ----------
        elem_ratio : str
            Element ratio to compute of form ``elem1/elem2``

        Returns
        -------
        float:
            Element ratio
        """

        element_dict = self.compute_elements_mix()
        elem1, elem2 = elem_ratio.split("/")

        if elem1 not in element_dict:
            self.error(f"None of the gases have the element {elem1}")
            raise ValueError(f"No gas has element {elem1}")
        if elem2 not in element_dict:
            self.error(f"None of the gases have the element {elem2}")
            raise ValueError(f"No gas has element {elem2}")

        return element_dict[elem1].sum() / element_dict[elem2].sum()

    def fitting_parameters(self) -> t.Dict[str, FittingType]:
        """Overrides the fitting parameters to return
        one with all the gas profile parameters as well

        Returns
        -------

        fit_param:
            Dictionary of fitting parameters

        """
        full_dict = {}
        for gas in self._gases:
            full_dict.update(gas.fitting_parameters())

        full_dict.update(self._param_dict)

        return full_dict

    def initialize_chemistry(
        self,
        nlayers: int,
        temperature_profile: npt.NDArray[np.float64],
        pressure_profile: npt.NDArray[np.float64],
        altitude_profile: t.Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        """Initialize the chemistry.

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
        self.info("Initializing chemistry model")

        mix_profile = []

        for gas in self._gases:
            gas.initialize_profile(
                nlayers, temperature_profile, pressure_profile, altitude_profile
            )
            mix_profile.append(gas.mixProfile)

        total_mix = sum(mix_profile)

        self.debug("Total mix output %s", total_mix)

        validity = np.any(total_mix > 1.0)

        self.debug("Is invalid? %s", validity)

        if validity:
            self.error("Greater than 1.0 chemistry profile detected")
            raise InvalidChemistryException

        mixratio_remainder = 1.0 - total_mix

        mixratio_remainder += np.zeros(shape=(nlayers))
        mix_profile = self.fill_atmosphere(mixratio_remainder) + mix_profile

        if len(mix_profile) > 0:
            self._mix_profile = np.vstack(mix_profile)
        else:
            self._mix_profile = 0.0

        super().initialize_chemistry(
            nlayers, temperature_profile, pressure_profile, altitude_profile
        )

        super().compute_mu_profile(nlayers)

    def fill_atmosphere(self, mixratio_remainder: float) -> t.List[float]:
        """Fills the atmosphere with the fill gases."""
        fill = []

        if len(self._fill_gases) == 1:
            return [mixratio_remainder]
        else:
            main_molecule = mixratio_remainder * (1 / (1 + sum(self._fill_ratio)))

            fill.append(main_molecule)
            for _, ratio in zip(self._fill_gases[1:], self._fill_ratio):
                second_molecule = ratio * main_molecule
                fill.append(second_molecule)
        return fill

    # @property
    # def activeGasMixProfile(self):
    #     """
    #     Active gas layer by layer mix profile

    #     Returns
    #     -------
    #     active_mix_profile : :obj:`array`

    #     """
    #     return self.mixProfile[self._active_mask]

    # @property
    # def inactiveGasMixProfile(self):
    #     """
    #     Inactive gas layer by layer mix profile

    #     Returns
    #     -------
    #     inactive_mix_profile : :obj:`array`

    #     """
    #     return self.mixProfile[self._inactive_mask]

    def write(self, output: OutputGroup) -> OutputGroup:
        """Write chemistry to output group."""
        gas_entry = super().write(output)
        if isinstance(self._fill_gases, float):
            gas_entry.write_scalar("ratio", self._fill_ratio)
        elif hasattr(self._fill_gases, "__len__"):
            gas_entry.write_array("ratio", np.array(self._fill_ratio))
        gas_entry.write_string_array("fill_gases", self._fill_gases)
        for gas in self._gases:
            gas.write(gas_entry)

        return gas_entry

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Input keywords for free chemistry."""
        return (
            "taurex",
            "free",
        )

    def citations(self):
        """Collect and return citations from all gases."""
        from taurex.data.citation import unique_citations_only

        old = super().citations()
        for g in self._gases:
            old.extend(g.citations())

        return unique_citations_only(old)

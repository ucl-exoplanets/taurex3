"""Basic forward model."""

import typing as t

import numpy as np
import numpy.typing as npt
from astropy import units as u

from taurex.binning import Binner
from taurex.chemistry import Chemistry
from taurex.output import OutputGroup
from taurex.planet import Planet
from taurex.pressure import PressureProfile
from taurex.stellar import Star
from taurex.temperature import TemperatureProfile
from taurex.types import ModelOutputType
from taurex.util import clip_native_to_wngrid

from .model import ForwardModel

if t.TYPE_CHECKING:
    from taurex.contributions import Contribution
else:
    Contribution = object


class SimpleForwardModel(ForwardModel):
    """A 'simple' 1D base model.

    It in the sense that its just
    a fairly standard single profiles model.
    It will handle settingup and initializing, building
    collecting parameters from given profiles for you.
    The only method that requires implementation is:

    - :func:`path_integral`

    """

    WARN = True

    def __init__(
        self,
        name: str,
        planet: t.Optional[Planet] = None,
        star: t.Optional[Star] = None,
        pressure_profile: t.Optional[PressureProfile] = None,
        temperature_profile: t.Optional[TemperatureProfile] = None,
        chemistry: t.Optional[Chemistry] = None,
        nlayers: t.Optional[int] = 100,
        atm_min_pressure: t.Optional[float] = 1e-4,
        atm_max_pressure: t.Optional[float] = 1e6,
        contributions: t.Optional[t.List[Contribution]] = None,
    ):
        """Initialize a 1D forward model.


        Parameters
        ----------
        name: str
            Name to use in logging

        planet:
            Planet model, default planet is Jupiter

        star:
            Star model, default star is Sun-like

        pressure_profile:
            Pressure model, alternative is to set ``nlayers``, ``atm_min_pressure``
            and ``atm_max_pressure``

        temperature_profile:
            Temperature model, default is an
            :class:`~taurex.data.profiles.temperature.isothermal.Isothermal`
            profile at 1500 K

        chemistry:
            Chemistry model, default is
            :class:`~taurex.data.profiles.chemistry.taurexchemistry.TaurexChemistry`
            with ``H2O`` and ``CH4``

        nlayers: int, optional
            Number of layers. Used if ``pressure_profile`` is not defined.

        atm_min_pressure: float, optional
            Pressure at TOA. Used if ``pressure_profile`` is not defined.

        atm_max_pressure: float, optional
            Pressure at BOA. Used if ``pressure_profile`` is not defined.

        """
        import warnings

        super().__init__(name)

        self._planet = planet
        self._star = star
        self._pressure_profile = pressure_profile
        self._temperature_profile = temperature_profile
        self._chemistry = chemistry
        self.debug(
            "Passed: %s %s %s %s %s",
            planet,
            star,
            pressure_profile,
            temperature_profile,
            chemistry,
        )
        self.altitude_profile = None
        self.scaleheight_profile = None
        self.gravity_profile = None
        self._setup_defaults(nlayers, atm_min_pressure, atm_max_pressure)

        self._initialized = False

        self._sigma_opacities = None

        self._native_grid = None

        if contributions:
            for contrib in contributions:
                self.add_contribution(contrib)

        if self.WARN:
            warnings.warn(
                "SimpleForwardModel is deprecated. Use OneDForwardModel instead",
                DeprecationWarning,
                stacklevel=2,
            )

        self.built = False

    def _compute_inital_mu(self):
        """Compute an initial molecular profile."""
        import warnings

        from taurex.data.profiles.chemistry import ConstantGas, TaurexChemistry

        warnings.warn(
            "This method is deprecated and will be removed in a "
            "future version as altitude is never needed.",
            DeprecationWarning,
            stacklevel=2,
        )

        tc = TaurexChemistry()
        tc.addGas(ConstantGas("H2O"))
        self._inital_mu = tc

    def _setup_defaults(
        self, nlayers: int, atm_min_pressure: float, atm_max_pressure: float
    ) -> None:
        """Setup default profiles if not defined.

        Parameters
        ----------
        nlayers: int
            Number of layers

        atm_min_pressure: float
            Pressure at TOA

        atm_max_pressure: float
            Pressure at BOA




        """
        if self._pressure_profile is None:
            from taurex.data.profiles.pressure import SimplePressureProfile

            self.info(
                "No pressure profile defined, using simple pressure " "profile with"
            )
            self.info(
                "parameters nlayers: %s, atm_pressure_range=(%s,%s)",
                nlayers,
                atm_min_pressure,
                atm_max_pressure,
            )

            self._pressure_profile = SimplePressureProfile(
                nlayers, atm_min_pressure, atm_max_pressure
            )

        if self._planet is None:
            from taurex.data import Planet

            self.warning("No planet defined, using Jupiter as planet")
            self._planet = Planet()

        if self._temperature_profile is None:
            from taurex.data.profiles.temperature import Isothermal

            self.warning(
                "No temeprature profile defined using default "
                "Isothermal profile with T=1500 K"
            )
            self._temperature_profile = Isothermal()

        if self._chemistry is None:
            from taurex.data.profiles.chemistry import ConstantGas, TaurexChemistry

            tc = TaurexChemistry()
            self.warning(
                "No gas profile set, using constant profile with H2O " "and CH4"
            )
            tc.addGas(ConstantGas("H2O", mix_ratio=1e-5))
            tc.addGas(ConstantGas("CH4", mix_ratio=1e-6))
            self._chemistry = tc

        if self._star is None:
            from taurex.data.stellar import BlackbodyStar

            self.warning("No star, using the Sun")
            self._star = BlackbodyStar()

    def initialize_profiles(self) -> None:
        """Initializes all profiles."""
        self.info("Computing pressure profile")

        self.pressure.compute_pressure_profile()

        self._temperature_profile.initialize_profile(
            self._planet, self.pressure.nLayers, self.pressure.profile
        )

        # Initialize the atmosphere with a constant gas profile
        # if self._initialized is False:
        #     self._inital_mu.initialize_chemistry(
        #         self.pressure.nLayers,
        #         self.temperatureProfile,
        #         self.pressureProfile,
        #         None,
        #     )

        #     self._compute_altitude_gravity_scaleheight_profile(
        #         self._inital_mu.muProfile
        #     )

        #     self._initialized = True

        # Setup any photochemistry
        self._chemistry.set_star_planet(self.star, self.planet)

        # Now initialize the gas profile real
        self._chemistry.initialize_chemistry(
            self.pressure.nLayers,
            self.temperatureProfile,
            self.pressureProfile,
            None,
            # self.altitude_profile,
        )

        # Compute gravity scale height
        self._compute_altitude_gravity_scaleheight_profile()

    def collect_fitting_parameters(self) -> None:
        """Combine all fitting parameters from components."""
        self._fitting_parameters = {}
        self._fitting_parameters.update(self.fitting_parameters())
        self._fitting_parameters.update(self._planet.fitting_parameters())
        if self._star is not None:
            self._fitting_parameters.update(self._star.fitting_parameters())
        self._fitting_parameters.update(self.pressure.fitting_parameters())

        self._fitting_parameters.update(self._temperature_profile.fitting_parameters())

        self._fitting_parameters.update(self._chemistry.fitting_parameters())

        for contrib in self.contribution_list:
            self._fitting_parameters.update(contrib.fitting_parameters())

        self.debug(
            "Available Fitting params: %s", list(self._fitting_parameters.keys())
        )

    def collect_derived_parameters(self):
        """Combine all derived parameters from all components."""
        self._derived_parameters = {}
        self._derived_parameters.update(self.derived_parameters())
        self._derived_parameters.update(self._planet.derived_parameters())
        if self._star is not None:
            self._derived_parameters.update(self._star.derived_parameters())
        self._derived_parameters.update(self.pressure.derived_parameters())

        self._derived_parameters.update(self._temperature_profile.derived_parameters())

        self._derived_parameters.update(self._chemistry.derived_parameters())

        for contrib in self.contribution_list:
            self._derived_parameters.update(contrib.derived_parameters())

        self.debug(
            "Available derived params: %s", list(self._derived_parameters.keys())
        )

    def build(self) -> None:
        """Build the forward model.

        Must be called at least once before running :func:`model`

        Will automatically be called by :func:`model` if not called before.
        """

        self.contribution_list.sort(key=lambda x: x.order)

        self.info("Building model........")
        # self._compute_inital_mu()
        self.info("Collecting paramters")
        self.collect_fitting_parameters()
        self.collect_derived_parameters()
        self.info("Setting up profiles")
        # self.initialize_profiles()

        self.info("Setting up contributions")
        # for contrib in self.contribution_list:
        #     contrib.build(self)
        self.info("DONE")
        self.built = True

    # altitude, gravity and scale height profile
    def _compute_altitude_gravity_scaleheight_profile(
        self, mu_profile: t.Optional[npt.NDArray[np.float64]] = None
    ) -> None:
        """Computes altitude, gravity and scale height of the atmosphere.
        Only call after :func:`build` has been called at least once.

        Parameters
        ----------
        mu_profile, optional:
            Molecular weight profile at each layer

        """

        # from taurex.constants import KBOLTZ
        if mu_profile is None:
            mu_profile = self._chemistry.muProfile

        pressure_levels = self.pressure.pressure_profile_levels
        z, scaleheight, g, deltaz = self.planet.calculate_scale_properties(
            self.temperatureProfile, pressure_levels, mu_profile
        )
        self.altitude_profile = z[:-1]
        self.scaleheight_profile = scaleheight[:-1]
        self.gravity_profile = g[:-1]
        self.altitude_boundaries = z
        self.deltaz = deltaz

    @property
    def pressureProfile(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Central atmospheric pressure profile in Pa."""
        return self.pressure.profile

    @property
    def temperatureProfile(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Atmospheric temperature profile in K."""
        return self._temperature_profile.profile

    @property
    def densityProfile(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Atmospheric density profile in m-3."""
        from taurex.constants import KBOLTZ

        return (self.pressureProfile) / (KBOLTZ * self.temperatureProfile)

    @property
    def altitudeProfile(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Atmospheric altitude profile in m."""
        return self.altitude_profile

    @property
    def nLayers(self) -> int:  # noqa: N802
        """Number of layers."""
        return self.pressure.nLayers

    @property
    def chemistry(self) -> Chemistry:
        """Chemistry model."""
        return self._chemistry

    @property
    def pressure(self) -> PressureProfile:
        """Pressure model."""
        return self._pressure_profile

    @property
    def temperature(self) -> TemperatureProfile:
        """Temperature model."""
        return self._temperature_profile

    @property
    def star(self) -> Star:
        """Star model."""
        return self._star

    @property
    def planet(self) -> Planet:
        """Planet model."""
        return self._planet

    def set_native_grid(
        self, spectral_grid: t.Union[u.Quantity, npt.NDArray[np.float64]]
    ) -> None:
        """Sets the native grid.

        Parameters
        ----------
        wngrid:
            Wavenumber grid
        """
        if isinstance(spectral_grid, u.Quantity):
            wngrid = spectral_grid.to(u.k, equivalencies=u.spectral()).value

        wngrid = np.array(wngrid)
        # Sort the grid
        wngrid = np.sort(wngrid)

        self._native_grid = wngrid

    def auto_grid(self) -> None:
        """Automatically sets the native grid."""
        self._native_grid = None

    @property
    def nativeWavenumberGrid(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Best grid for given cross-sections.

        Searches through active molecules to determine the
        native wavenumber grid

        Returns
        -------

        wngrid: :obj:`array`
            Native grid

        Raises
        ------
        InvalidModelException
            If no active molecules in atmosphere
        """
        from taurex.cache import GlobalCache
        from taurex.cache.opacitycache import OpacityCache
        from taurex.exceptions import InvalidModelException

        if self._native_grid is not None:
            return self._native_grid
        cacher = OpacityCache()

        if GlobalCache()["opacity_method"] == "ktables":
            from taurex.cache.ktablecache import KTableCache

            cacher = KTableCache()

        active_gases = self.chemistry.activeGases

        wavenumbergrid = [cacher[gas].wavenumberGrid for gas in active_gases]

        current_grid = None
        for wn in wavenumbergrid:
            if current_grid is None:
                current_grid = wn
            if wn.shape[0] > current_grid.shape[0]:
                current_grid = wn

        if current_grid is None:
            self.error("No active molecules detected")
            self.error("Most likely no cross-sections/ktables were detected")
            raise InvalidModelException("No active absorbing molecules")

        return current_grid

    def model(
        self,
        wngrid: t.Optional[npt.NDArray[np.float64]] = None,
        cutoff_grid: t.Optional[bool] = True,
    ) -> ModelOutputType:
        """Runs the forward model.

        Parameters
        ----------

        wngrid:
            Wavenumber grid, default is to use native grid

        cutoff_grid:
            Run model only on ``wngrid`` given, default is ``True``

        Returns
        -------

        native_grid:
            Native wavenumber grid, clipped if ``wngrid`` passed

        depth:
            Resulting depth

        tau:
            Optical depth.

        extra: ``None``
            Empty
        """
        if not self.built:
            self.build()
        # Setup profiles
        self.initialize_profiles()

        # Clip grid if necessary
        native_grid = self.nativeWavenumberGrid
        if wngrid is not None and cutoff_grid:
            native_grid = clip_native_to_wngrid(native_grid, wngrid)

        # Initialize star
        self._star.initialize(native_grid)

        # Prepare contributions
        for contrib in self.contribution_list:
            contrib.prepare(self, native_grid)

        # Compute path integral
        absorp, tau = self.path_integral(native_grid, False)

        return native_grid, absorp, tau, None

    def model_contrib(
        self,
        wngrid: t.Optional[npt.NDArray[np.float64]] = None,
        cutoff_grid: t.Optional[bool] = True,
    ) -> t.Tuple[
        npt.NDArray[np.float64],
        t.Dict[
            str,
            t.Tuple[
                npt.NDArray[np.float64], npt.NDArray[np.float64], t.Optional[t.Any]
            ],
        ],
    ]:
        """Models each contribution seperately."""
        # Setup profiles
        self.initialize_profiles()

        # Copy over contribution list
        full_contrib_list = self.contribution_list
        # Get the native grid
        native_grid = self.nativeWavenumberGrid

        # Clip grid
        all_contrib_dict = {}
        if wngrid is not None and cutoff_grid:
            native_grid = clip_native_to_wngrid(native_grid, wngrid)

        # Initialize star
        self._star.initialize(native_grid)

        for contrib in full_contrib_list:
            self.contribution_list = [contrib]
            contrib.prepare(self, native_grid)
            absorp, tau = self.path_integral(native_grid, False)
            all_contrib_dict[contrib.name] = (absorp, tau, None)

        self.contribution_list = full_contrib_list
        return native_grid, all_contrib_dict

    def model_full_contrib(
        self,
        wngrid: t.Optional[npt.NDArray[np.float64]] = None,
        cutoff_grid: t.Optional[bool] = True,
    ) -> t.Tuple[
        npt.NDArray[np.float64],
        t.Dict[
            str,
            t.List[
                t.Tuple[
                    str,
                    npt.NDArray[np.float64],
                    npt.NDArray[np.float64],
                    t.Optional[t.Any],
                ]
            ],
        ],
    ]:
        """Model each sub-component of each contribution.

        Like :func:`model_contrib` except all components for
        each contribution are modelled

        e.g. Absorption will be modelled seperately for each molecule
        in the atmosphere.

        """
        native_grid = self.nativeWavenumberGrid
        if wngrid is not None and cutoff_grid:
            native_grid = clip_native_to_wngrid(native_grid, wngrid)

        self.initialize_profiles()
        self._star.initialize(native_grid)

        result_dict = {}

        full_contrib_list = self.contribution_list

        self.debug("NATIVE GRID %s", native_grid.shape)

        self.info("Modelling each contribution.....")
        for contrib in full_contrib_list:
            self.contribution_list = [contrib]
            contrib_name = contrib.name
            contrib_res_list = []

            for name, __ in contrib.prepare_each(self, native_grid):
                self.info("\t%s---%s contribtuion", contrib_name, name)
                absorp, tau = self.path_integral(native_grid, False)
                contrib_res_list.append((name, absorp, tau, None))

            result_dict[contrib_name] = contrib_res_list

        self.contribution_list = full_contrib_list
        return native_grid, result_dict

    def compute_error(
        self,
        samples: t.Callable[[], float],
        wngrid: t.Optional[npt.NDArray[np.float64]] = None,
        binner: t.Optional[Binner] = None,
    ) -> t.Tuple[
        t.Dict[str, npt.NDArray[np.float64]], t.Dict[str, npt.NDArray[np.float64]]
    ]:
        """Computes standard deviations from samples.

        Parameters
        ----------

        samples:
            A callable function that returns a weight for each sample

        wngrid:
            Wavenumber grid, default is to use native grid

        binner:
            A :class:`~taurex.binning.binner.Binner` object to bin the spectrum

        """
        from taurex.util.math import OnlineVariance

        tp_profiles = OnlineVariance()
        active_gases = OnlineVariance()
        inactive_gases = OnlineVariance()
        has_condensates = self.chemistry.hasCondensates

        cond = OnlineVariance() if has_condensates else None

        binned_spectrum = OnlineVariance() if binner is not None else None
        native_spectrum = OnlineVariance()

        for weight in samples():
            native_grid, native, tau, _ = self.model(wngrid=wngrid, cutoff_grid=False)

            tp_profiles.update(self.temperatureProfile, weight=weight)
            active_gases.update(self.chemistry.activeGasMixProfile, weight=weight)
            inactive_gases.update(self.chemistry.inactiveGasMixProfile, weight=weight)

            if cond is not None:
                cond.update(self.chemistry.condensateMixProfile, weight=weight)

            native_spectrum.update(
                np.maximum(np.nan_to_num(native), 1e-20), weight=weight
            )

            if binned_spectrum is not None:
                binned = np.maximum(
                    np.nan_to_num(binner.bindown(native_grid, native)[1]), 1e-20
                )

                binned_spectrum.update(binned, weight=weight)

        tp_std = np.sqrt(tp_profiles.parallelVariance())
        active_std = np.sqrt(active_gases.parallelVariance())
        inactive_std = np.sqrt(inactive_gases.parallelVariance())

        profile_dict = {
            "temp_profile_std": tp_std,
            "active_mix_profile_std": active_std,
            "inactive_mix_profile_std": inactive_std,
        }

        if cond is not None:
            profile_dict["condensate_profile_std"] = np.sqrt(cond.parallelVariance())

        spectrum_dict = {"native_std": np.sqrt(native_spectrum.parallelVariance())}
        if binned_spectrum is not None:
            spectrum_dict["binned_std"] = np.sqrt(binned_spectrum.parallelVariance())

        return profile_dict, spectrum_dict

    def path_integral(
        self, wngrid: npt.NDArray[np.float64], return_contrib: t.Optional[bool]
    ) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Main integration function.

        Must return the absorption and optical depth for the given wngrid.

        """
        raise NotImplementedError

    def generate_profiles(self) -> None:
        """Generate profiles to store."""
        from taurex.util.output import generate_profile_dict

        prof = generate_profile_dict(self)
        prof["mu_profile"] = self.chemistry.muProfile
        return prof

    def write(self, output: OutputGroup) -> OutputGroup:
        """Write forward model to output group.

        Will also write all components to the output group.

        """
        # Run a model if needed
        self.model()

        model = super().write(output)

        self._chemistry.write(model)
        self._temperature_profile.write(model)
        self.pressure.write(model)
        self._planet.write(model)
        self._star.write(model)
        return model

    def citations(self) -> str:
        """Citations for model."""
        from taurex.cache import OpacityCache
        from taurex.data.citation import unique_citations_only

        model_citations = super().citations()
        model_citations.extend(self.chemistry.citations())
        model_citations.extend(self.temperature.citations())
        model_citations.extend(self.pressure.citations())
        model_citations.extend(self.planet.citations())
        model_citations.extend(self.star.citations())

        # Get cache citations
        active_gases = self.chemistry.activeGases

        opacitycache = OpacityCache()

        # Do cross sections

        for g in active_gases:
            try:
                xsec = opacitycache.opacity_dict[g]
                model_citations.extend(xsec.citations())
            except KeyError:
                continue
        return unique_citations_only(model_citations)


# A much better alias for the class
class OneDForwardModel(SimpleForwardModel):
    """A forward model for a 1D atmosphere.

    Automatically sets up the following profiles:
        - Temperature
        - Pressure
        - Chemistry
        - Planet
        - Star
        - Contributions


    Must implement the following methods:
        - :func:`path_integral`
            - Main integration function that must
            return the absorption and optical depth
            for the given wngrid.

    """

    pass

"""Handles loading of classes from plugins and built-in classes"""

# flake8: noqa:N802

import importlib.metadata
import inspect
import pathlib
import sys
import typing as t
from types import ModuleType

from taurex.core import Singleton
from taurex.log import Logger
from taurex.mixin import Mixin
from taurex.types import PathLike

try:
    EPType = importlib.metadata.EntryPoints
except AttributeError:
    EPType = t.Any


def entry_points(*, group: str) -> EPType:  # type: ignore[name-defined]
    """Wrapper for entry_points to support Python 3.8+ instead of 3.10+."""
    if sys.version_info >= (3, 10):
        return importlib.metadata.entry_points(group=group)
    epg = importlib.metadata.entry_points()

    return epg.get(group, [])


class InputKeywordMethod(t.Protocol):
    """Protocol for classes that have input_keywords()"""

    @classmethod
    def input_keywords(cls) -> t.Sequence[str]:
        """Returns a list of input keywords"""
        ...


class DiscoverMethod(t.Protocol):
    """Protocol for classes that have discover() methods"""

    @classmethod
    def discover(cls) -> t.List[t.Tuple[str, t.Any]]:
        """Returns a list of input keywords"""
        ...


T = t.TypeVar("T", bound=InputKeywordMethod)
AllT = t.TypeVar("AllT")
OpacT = t.TypeVar("OpacT", bound=DiscoverMethod)
MixinT = t.TypeVar("MixinT", bound=Mixin)


class ClassFactory(Singleton):
    """A singleton factory the discovers new classes from plugins."""

    def init(self) -> None:
        self.log = Logger("ClassFactory")

        self.extension_paths = []
        self.reload_plugins()

    def set_extension_paths(
        self,
        paths: t.Optional[t.List[PathLike]] = None,
        reload: t.Optional[bool] = True,
    ) -> None:
        self.extension_paths = [pathlib.Path(p) for p in paths] if paths else []
        if reload:
            self.reload_plugins()

    def reload_plugins(self) -> None:
        """Reload all plugins and built-in classes."""
        self.log.info("Reloading all modules and plugins")
        self.setup_batteries_included()
        self.setup_batteries_included_mixin()
        self.load_plugins()
        self.load_extension_paths()

    def setup_batteries_included_mixin(self):
        """Collect all the mixin classes that are built into TauREx 3."""
        import taurex.mixin as mixins

        self._temp_mixin_klasses = set(self._collect_temperatures_mixin(mixins))
        self._chem_mixin_klasses = set(self._collect_chemistry_mixin(mixins))
        self._gas_mixin_klasses = set(self._collect_gas_mixin(mixins))
        self._press_mixin_klasses = set(self._collect_pressure_mixin(mixins))
        self._planet_mixin_klasses = set(self._collect_planets_mixin(mixins))
        self._star_mixin_klasses = set(self._collect_star_mixin(mixins))
        self._inst_mixin_klasses = set(self._collect_instrument_mixin(mixins))
        self._model_mixin_klasses = set(self._collect_model_mixin(mixins))
        self._obs_mixin_klasses = set(self._collect_observation_mixin(mixins))
        self._contrib_mixin_klasses = set(self._collect_contributions_mixin(mixins))
        self._opt_mixin_klasses = set(self._collect_optimizer_mixin(mixins))
        self._any_mixin_klasses = set(self._collect_classes(mixins, mixins.AnyMixin))

    def setup_batteries_included(self) -> None:
        """Collect all the classes that are built into TauREx 3."""
        from taurex import (
            chemistry,
            contributions,
            instruments,
            model,
            opacity,
            optimizer,
            planet,
            pressure,
            spectrum,
            stellar,
            temperature,
        )
        from taurex.core import priors
        from taurex.opacity import ktables

        self._temp_klasses = set(self._collect_temperatures(temperature))
        self._chem_klasses = set(self._collect_chemistry(chemistry))
        self._gas_klasses = set(self._collect_gas(chemistry))
        self._press_klasses = set(self._collect_pressure(pressure))
        self._planet_klasses = set(self._collect_planets(planet))
        self._star_klasses = set(self._collect_star(stellar))
        self._inst_klasses = set(self._collect_instrument(instruments))
        self._model_klasses = set(self._collect_model(model))
        self._obs_klasses = set(self._collect_observation(spectrum))
        self._contrib_klasses = set(self._collect_contributions(contributions))

        self._opt_klasses = set(self._collect_optimizer(optimizer))
        self._opac_klasses = set(self._collect_opacity(opacity))
        self._ktab_klasses = set(self._collect_ktables(ktables))
        self._prior_klasses = set(self._collect_priors(priors))

    def load_plugin(self, plugin_module: ModuleType) -> None:
        """Load a plugin module and collect all the classes from it."""
        self._temp_klasses.update(self._collect_temperatures(plugin_module))
        self._chem_klasses.update(self._collect_chemistry(plugin_module))
        self._gas_klasses.update(self._collect_gas(plugin_module))
        self._press_klasses.update(self._collect_pressure(plugin_module))
        self._planet_klasses.update(self._collect_planets(plugin_module))
        self._star_klasses.update(self._collect_star(plugin_module))
        self._inst_klasses.update(self._collect_instrument(plugin_module))
        self._model_klasses.update(self._collect_model(plugin_module))
        self._obs_klasses.update(self._collect_observation(plugin_module))
        self._contrib_klasses.update(self._collect_contributions(plugin_module))
        self._opt_klasses.update(self._collect_optimizer(plugin_module))
        self._opac_klasses.update(self._collect_opacity(plugin_module))
        self._prior_klasses.update(self._collect_priors(plugin_module))
        self._ktab_klasses.update(self._collect_ktables(plugin_module))

        self._temp_mixin_klasses.update(self._collect_temperatures_mixin(plugin_module))
        self._chem_mixin_klasses.update(self._collect_chemistry_mixin(plugin_module))
        self._gas_mixin_klasses.update(self._collect_gas_mixin(plugin_module))
        self._press_mixin_klasses.update(self._collect_pressure_mixin(plugin_module))
        self._planet_mixin_klasses.update(self._collect_planets_mixin(plugin_module))
        self._star_mixin_klasses.update(self._collect_star_mixin(plugin_module))
        self._inst_mixin_klasses.update(self._collect_instrument_mixin(plugin_module))
        self._model_mixin_klasses.update(self._collect_model_mixin(plugin_module))
        self._obs_mixin_klasses.update(self._collect_observation_mixin(plugin_module))
        self._contrib_mixin_klasses.update(
            self._collect_contributions_mixin(plugin_module)
        )

        self._opt_mixin_klasses.update(self._collect_optimizer_mixin(plugin_module))

    def discover_plugins(self) -> t.Tuple[t.Dict[str, ModuleType], t.Dict[str, str]]:
        """Discover all the plugins that are available from entry points."""

        plugins = {}
        failed_plugins = {}

        plugins_entrypoint = entry_points(group="taurex.plugins")

        for entry_point in plugins_entrypoint:
            entry_point_name = entry_point.name

            try:
                module: ModuleType = entry_point.load()
            except Exception as e:
                # For whatever reason do not attempt to load the plugin
                self.log.warning("Could not load plugin %s", entry_point_name)
                self.log.warning("Reason: %s", str(e))
                failed_plugins[entry_point_name] = str(e)
                continue

            plugins[entry_point_name] = module

        return plugins, failed_plugins

    def load_plugins(self):
        """Load all the plugins that are available."""
        plugins, failed_plugins = self.discover_plugins()
        self.log.info("----------Plugin loading---------")
        self.log.debug("Discovered plugins %s", plugins.values())

        for k, v in plugins.items():
            self.log.info("Loading %s", k)
            self.load_plugin(v)

    def load_extension_paths(self):
        """Load all the plugins from the extension paths."""
        import importlib.util

        paths = self.extension_paths
        if paths:
            # Make sure they're unique
            all_files = set(
                sum(
                    [list(p.glob("*.py")) for p in paths],
                    [],
                )
            )
            for f in all_files:
                self.log.info("Loading extensions from %s", f)
                module_name = f.stem
                spec = importlib.util.spec_from_file_location(module_name, f)
                foo = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(foo)
                    self.load_plugin(foo)
                except Exception as e:
                    self.log.warning("Could not load extension from file %s", f)
                    self.log.warning("Reason: %s", str(e))

    def _collect_classes(
        self,
        module: ModuleType,
        base_klass: t.Type[t.Union[T, OpacT, MixinT]],
        check_validity: t.Optional[bool] = True,
    ) -> t.List[t.Type[t.Union[T, OpacT, MixinT]]]:
        """Collects all classes that are a subclass of base."""
        klasses = []
        clsmembers = inspect.getmembers(module, inspect.isclass)
        for _, c in clsmembers:
            if issubclass(c, base_klass) and (c is not base_klass):
                self.log.debug(f" Found class {c.__name__}")
            else:
                continue

            if check_validity:
                try:
                    if hasattr(c, "discover"):
                        c.discover()
                    if hasattr(c, "input_keywords"):
                        c.input_keywords()
                    klasses.append(c)
                except (NotImplementedError, AttributeError):
                    self.log.debug(
                        f"Class {c.__name__} does not implement input_keywords"
                    )
                    continue
            else:
                klasses.append(c)

        return klasses

    def _collect_temperatures(self, module: ModuleType):
        from taurex.temperature import TemperatureProfile

        return self._collect_classes(module, TemperatureProfile)

    def _collect_chemistry(self, module: ModuleType):
        from taurex.chemistry import Chemistry

        return self._collect_classes(module, Chemistry)

    def _collect_gas(self, module: ModuleType):
        from taurex.chemistry import Gas

        return self._collect_classes(module, Gas)

    def _collect_pressure(self, module: ModuleType):
        from taurex.pressure import PressureProfile

        return self._collect_classes(module, PressureProfile)

    def _collect_planets(self, module: ModuleType):
        from taurex.planet import BasePlanet

        return self._collect_classes(module, BasePlanet)

    def _collect_star(self, module: ModuleType):
        from taurex.stellar import Star

        return self._collect_classes(module, Star)

    def _collect_instrument(self, module: ModuleType):
        from taurex.instruments import Instrument

        return self._collect_classes(module, Instrument)

    def _collect_model(self, module: ModuleType):
        from taurex.model import ForwardModel, OneDForwardModel, SimpleForwardModel

        return self._collect_classes(module, ForwardModel)

    def _collect_contributions(self, module):
        from taurex.contributions import Contribution

        return self._collect_classes(module, Contribution)

    def _collect_optimizer(self, module):
        from taurex.optimizer import Optimizer

        return self._collect_classes(module, Optimizer)

    def _collect_opacity(self, module):
        from taurex.opacity import Opacity

        return [c for c in self._collect_classes(module, Opacity)]

    def _collect_ktables(self, module):
        from taurex.opacity.ktables import KTable

        return [c for c in self._collect_classes(module, KTable)]

    def _collect_observation(self, module):
        from taurex.spectrum import BaseSpectrum

        return [c for c in self._collect_classes(module, BaseSpectrum)]

    def _collect_priors(self, module):
        from taurex.core.priors import Prior

        # Does not need to be valid since it does not have input_keywords or discover
        return self._collect_classes(module, Prior, check_validity=False)

    # Mixins
    def _collect_temperatures_mixin(self, module):
        from taurex.mixin import TemperatureMixin

        return self._collect_classes(module, TemperatureMixin)

    def _collect_chemistry_mixin(self, module):
        from taurex.mixin import ChemistryMixin

        return self._collect_classes(module, ChemistryMixin)

    def _collect_gas_mixin(self, module):
        from taurex.mixin import GasMixin

        return self._collect_classes(module, GasMixin)

    def _collect_pressure_mixin(self, module):
        from taurex.mixin import PressureMixin

        return self._collect_classes(module, PressureMixin)

    def _collect_planets_mixin(self, module):
        from taurex.mixin import PlanetMixin

        return self._collect_classes(module, PlanetMixin)

    def _collect_star_mixin(self, module):
        from taurex.mixin import StarMixin

        return self._collect_classes(module, StarMixin)

    def _collect_instrument_mixin(self, module):
        from taurex.mixin import InstrumentMixin

        return self._collect_classes(module, InstrumentMixin)

    def _collect_model_mixin(self, module):
        from taurex.mixin import ForwardModelMixin

        return self._collect_classes(module, ForwardModelMixin)

    def _collect_contributions_mixin(self, module):
        from taurex.mixin import ContributionMixin

        return self._collect_classes(module, ContributionMixin)

    def _collect_optimizer_mixin(self, module):
        from taurex.mixin import OptimizerMixin

        return self._collect_classes(module, OptimizerMixin)

    def _collect_observation_mixin(self, module):
        from taurex.mixin import ObservationMixin

        return self._collect_classes(module, ObservationMixin)

    @property
    def class_dict(self) -> t.Dict[t.Type[t.Any], t.Set[t.Type[t.Any]]]:
        """Returns a dictionary of all classes that are available in TauREx 3."""
        from taurex.chemistry import Chemistry, Gas
        from taurex.contributions import Contribution
        from taurex.core.priors import Prior
        from taurex.instruments import Instrument
        from taurex.model import ForwardModel
        from taurex.opacity import Opacity
        from taurex.opacity.ktables import KTable
        from taurex.optimizer import Optimizer
        from taurex.planet import BasePlanet
        from taurex.pressure import PressureProfile
        from taurex.spectrum import BaseSpectrum
        from taurex.stellar import Star
        from taurex.temperature import TemperatureProfile

        klass_dict = {
            TemperatureProfile: self.temperatureKlasses,
            Chemistry: self.chemistryKlasses,
            Gas: self.gasKlasses,
            PressureProfile: self.pressureKlasses,
            BasePlanet: self.planetKlasses,
            Star: self.starKlasses,
            Instrument: self.instrumentKlasses,
            ForwardModel: self.modelKlasses,
            Contribution: self.contributionKlasses,
            Optimizer: self.optimizerKlasses,
            Opacity: self.opacityKlasses,
            KTable: self.ktableKlasses,
            BaseSpectrum: self.observationKlasses,
            Prior: self.priorKlasses,
        }

    def list_from_base(self, klass_type: t.Type[t.Any]) -> t.List[t.Type[t.Any]]:
        """Returns a list of classes that are a subclass of klass_type."""

        return self.class_dict[klass_type]

    @property
    def temperatureKlasses(self):
        """Returns a list of all temperature classes."""
        return self._temp_klasses

    @property
    def chemistryKlasses(self):
        """Returns a list of all chemistry classes."""
        return self._chem_klasses

    @property
    def gasKlasses(self):
        """Returns a list of all gas classes."""
        return self._gas_klasses

    @property
    def pressureKlasses(self):
        """Returns a list of all pressure classes."""
        return self._press_klasses

    @property
    def planetKlasses(self):
        """Returns a list of all planet classes."""
        return self._planet_klasses

    @property
    def starKlasses(self):
        """Returns a list of all star classes."""
        return self._star_klasses

    @property
    def instrumentKlasses(self):
        """Returns a list of all instrument classes."""
        return self._inst_klasses

    @property
    def modelKlasses(self):
        """Returns a list of all model classes."""
        return self._model_klasses

    @property
    def contributionKlasses(self):
        """Returns a list of all contribution classes."""
        return self._contrib_klasses

    @property
    def optimizerKlasses(self):
        """Returns a list of all optimizer classes."""
        return self._opt_klasses

    @property
    def opacityKlasses(self):
        """Returns a list of all opacity classes."""
        return self._opac_klasses

    @property
    def ktableKlasses(self):
        """Returns a list of all ktable classes."""
        return self._ktab_klasses

    @property
    def observationKlasses(self):
        """Returns a list of all observation classes."""
        return self._obs_klasses

    @property
    def priorKlasses(self):
        """Returns a list of all prior classes."""
        return self._prior_klasses

    # Mixins

    @property
    def temperatureMixinKlasses(self):
        """Returns a list of all temperature mixin classes."""
        return self._temp_mixin_klasses

    @property
    def chemistryMixinKlasses(self):
        """Returns a list of all chemistry mixin classes."""
        return self._chem_mixin_klasses

    @property
    def gasMixinKlasses(self):
        """Returns a list of all gas mixin classes."""
        return self._gas_mixin_klasses

    @property
    def pressureMixinKlasses(self):
        """Returns a list of all pressure mixin classes."""
        return self._press_mixin_klasses

    @property
    def planetMixinKlasses(self):
        """Returns a list of all planet mixin classes."""
        return self._planet_mixin_klasses

    @property
    def starMixinKlasses(self):
        """Returns a list of all star mixin classes."""
        return self._star_mixin_klasses

    @property
    def instrumentMixinKlasses(self):
        """Returns a list of all instrument mixin classes."""
        return self._inst_mixin_klasses

    @property
    def modelMixinKlasses(self):
        """Returns a list of all model mixin classes."""
        return self._model_mixin_klasses

    @property
    def contributionMixinKlasses(self):
        """Returns a list of all contribution mixin classes."""
        return self._contrib_mixin_klasses

    @property
    def optimizerMixinKlasses(self):
        """Returns a list of all optimizer mixin classes."""
        return self._opt_mixin_klasses

    @property
    def observationMixinKlasses(self):
        """Returns a list of all observation mixin classes."""
        return self._obs_mixin_klasses

    @property
    def all_klasses(self) -> t.Set[t.Union[T, OpacT, MixinT]]:
        """Returns a set of all classes that are available in TauREx 3."""
        return (
            self.temperatureKlasses
            | self.chemistryKlasses
            | self.gasKlasses
            | self.pressureKlasses
            | self.planetKlasses
            | self.starKlasses
            | self.instrumentKlasses
            | self.modelKlasses
            | self.contributionKlasses
            | self.optimizerKlasses
            | self.opacityKlasses
            | self.ktableKlasses
            | self.observationKlasses
            | self.priorKlasses
            | self.temperatureMixinKlasses
            | self.chemistryMixinKlasses
            | self.gasMixinKlasses
            | self.pressureMixinKlasses
            | self.planetMixinKlasses
            | self.starMixinKlasses
            | self.instrumentMixinKlasses
            | self.modelMixinKlasses
            | self.contributionMixinKlasses
            | self.optimizerMixinKlasses
            | self.observationMixinKlasses
        )

    @property
    def mixin_klasses(self) -> t.Set[t.Type[Mixin]]:
        """Returns a set of all mixin classes that are available in TauREx 3."""
        return (
            self.temperatureMixinKlasses
            | self.chemistryMixinKlasses
            | self.gasMixinKlasses
            | self.pressureMixinKlasses
            | self.planetMixinKlasses
            | self.starMixinKlasses
            | self.instrumentMixinKlasses
            | self.modelMixinKlasses
            | self.contributionMixinKlasses
            | self.optimizerMixinKlasses
            | self.observationMixinKlasses
        )

    def klass_from_base(self, base_class: AllT) -> t.Set[t.Type[AllT]]:
        """Returns a set of all classes that are subclass of base."""
        return {k for k in self.all_klasses if issubclass(k, base_class)}

    # def suitable_mixins(self, baseclass: AllT) -> t.Set[t.Type[Mixin]]:
    #     """Returns a set of all mixin classes that are suitable for mixin."""
    #     return {m for m in self.mixin_klasses if m.compatible(baseclass)}

    def find_klass(self, name: str) -> t.Optional[t.Union[T, OpacT, MixinT]]:
        """Returns a class with the given name."""
        for klass in self.all_klasses:
            if klass.__name__.lower() == name.lower():
                return klass
        return None

    def find_klass_from_keyword(
        self, keyword: str, baseclass: t.Optional[t.Type[T]] = None
    ) -> t.Type[T]:
        """Returns a class that has the given keyword.

        Returns the first found if multiple klasses have the same keyword.

        """
        klasses = (
            self.all_klasses if baseclass is None else self.klass_from_base(baseclass)
        )

        for k in klasses:
            if hasattr(k, "input_keywords") and keyword.lower() in [
                s.lower() for s in k.input_keywords()
            ]:
                return k
        raise KeyError(f"Could not find class with keyword {keyword}")

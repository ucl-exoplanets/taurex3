"""Core mixin functions."""

import typing as t

from taurex.chemistry import Chemistry, Gas
from taurex.contributions import Contribution
from taurex.instruments import Instrument
from taurex.log import setup_log
from taurex.model import ForwardModel
from taurex.optimizer import Optimizer
from taurex.planet import Planet
from taurex.pressure import PressureProfile
from taurex.spectrum import BaseSpectrum
from taurex.stellar import Star
from taurex.temperature import TemperatureProfile

from ..core import Citable, Fittable

if t.TYPE_CHECKING:
    # Useful for type checking but not for runtime
    _BaseStar = Star
    _BaseTemperatureProfile = TemperatureProfile
    _BasePressureProfile = PressureProfile
    _BasePlanet = Planet
    _BaseContribution = Contribution
    _BaseChemistry = Chemistry
    _BaseForwardModel = ForwardModel
    _BaseSpectrum = BaseSpectrum
    _BaseOptimizer = Optimizer
    _BaseInstrument = Instrument
    _BaseGas = Gas

else:
    _BaseStar = object
    _BaseTemperatureProfile = object
    _BasePressureProfile = object
    _BasePlanet = object
    _BaseContribution = object
    _BaseChemistry = object
    _BaseForwardModel = object
    _BaseSpectrum = object
    _BaseOptimizer = object
    _BaseInstrument = object
    _BaseGas = object


# Try and create __init_mixin__

_log = setup_log(__name__)

T = t.TypeVar("T")

M = t.TypeVar("M", bound="Mixin")


class MixinProtocol(t.Protocol):
    """Mixin protocol."""

    def __init_mixin__(self, **kwargs: t.Dict[str, t.Any]) -> None:
        ...


def mixed_init(self, **kwargs: t.Dict[str, t.Any]) -> None:
    """Initialisation function for mixed classes."""
    import inspect

    new_class = self.__class__
    base_class = self.__class__.__bases__[-1]
    args = list(inspect.signature(base_class.__init__).parameters.keys())

    # Remove self
    if "self" in args:
        args.remove("self")
    new_kwargs = {}
    for k, v in kwargs.items():
        if k in args:
            new_kwargs[k] = v

    for klass in reversed(new_class.__bases__):
        klass.__init__(self, **new_kwargs)

    new_kwargs = {}

    for klass in reversed(new_class.__bases__[:-1]):
        klass = t.cast(t.Type[Mixin], klass)
        args = list(inspect.signature(klass.__init_mixin__).parameters.keys())
        if "self" in args:
            args.remove("self")
        new_kwargs = {}
        for k, v in kwargs.items():
            if k in args:
                new_kwargs[k] = v
        klass.__init_mixin__(self, **new_kwargs)
    # Embed class bases into object


class Mixin(MixinProtocol, Fittable, Citable, t.Generic[T]):
    """Base mixin class."""

    KLASS_COMPAT: t.Type[T] = None

    def __init__(self, **kwargs) -> None:
        """Constructor.

        Should not be called directly.

        """
        old_fitting_parameters = {}
        old_derived_parameters = {}
        if hasattr(self, "_param_dict"):
            old_fitting_parameters = self._param_dict
            old_derived_parameters = self._derived_dict
        super().__init__(**kwargs)

        if not hasattr(self, "_param_dict"):
            self._param_dict = {}
            self._derived_dict = {}

        self._param_dict.update(old_fitting_parameters)
        self._derived_dict.update(old_derived_parameters)

    def __init_mixin__(self):
        """Main initialisation function for mixin.

        This should be implemented by the mixin class and not ``__init__``.

        """
        pass

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        raise NotImplementedError

    @classmethod
    def compatible(cls, other: t.Type) -> bool:
        if cls.KLASS_COMPAT:
            return issubclass(other, cls.KLASS_COMPAT)
        else:
            return False


class AnyMixin(Mixin[t.Any]):
    """Can enhance any class."""

    KLASS_COMPAT = object


class StarMixin(Mixin[Star], _BaseStar):
    """Enhances :class:`~taurex.data.stellar.star.Star`."""

    KLASS_COMPAT = Star


class TemperatureMixin(Mixin[TemperatureProfile], _BaseTemperatureProfile):
    """Enhances :class:`~taurex.data.profiles.temperature.TemperatureProfile`."""

    KLASS_COMPAT = TemperatureProfile


class PlanetMixin(Mixin[Planet], _BasePlanet):
    """Enhances :class:`~taurex.data.planet.Planet`."""

    KLASS_COMPAT = Planet


class ContributionMixin(Mixin[Contribution], _BaseContribution):
    """Enhances :class:`~taurex.contributions.Contribution`."""

    KLASS_COMPAT = Contribution


class ChemistryMixin(Mixin[Chemistry], _BaseChemistry):
    """Enhances :class:`~taurex.data.chemistry.Chemistry`."""

    KLASS_COMPAT = Chemistry


class PressureMixin(Mixin[PressureProfile], _BasePressureProfile):
    """Enhances :class:`~taurex.data.profiles.pressure.PressureProfile`."""

    KLASS_COMPAT = PressureProfile


class ForwardModelMixin(Mixin[ForwardModel], _BaseForwardModel):
    """Enhances :class:`~taurex.model.model.ForwardModel`."""

    KLASS_COMPAT = ForwardModel


class SpectrumMixin(Mixin[BaseSpectrum], _BaseSpectrum):
    """Enhances :class:`~taurex.spectrum.Spectrum`."""

    KLASS_COMPAT = BaseSpectrum


class ObservationMixin(SpectrumMixin):
    """Enhances :class:`~taurex.spectrum.Spectrum`."""

    pass


class OptimizerMixin(Mixin[Optimizer], _BaseOptimizer):
    """Enhances :class:`~taurex.optimizers.Optimizer`."""

    KLASS_COMPAT = Optimizer


class GasMixin(Mixin[Gas], _BaseGas):
    """Enhances :class:`~taurex.data.gas.Gas`."""

    KLASS_COMPAT = Gas


class InstrumentMixin(Mixin[Instrument], _BaseInstrument):
    """Enhances :class:`~taurex.instruments.instrument.Instrument`."""

    KLASS_COMPAT = Instrument


def determine_mixin_args(
    klasses: t.Sequence[t.Union[t.Type[T], t.Type[M]]]
) -> t.Tuple[t.Dict[str, t.Any], bool]:
    """Determine all arguments for a mixin class."""
    import inspect

    has_kvar = False
    all_kwargs = {}
    for klass in klasses:
        argspec = inspect.signature(klass.__init__)
        if issubclass(klass, Mixin):
            argspec = inspect.signature(klass.__init_mixin__)

        parameters = argspec.parameters
        args = list(parameters.keys())
        if "self" in args:
            args.remove("self")

        for arg in args:
            if parameters[arg].kind == inspect.Parameter.VAR_KEYWORD:
                has_kvar = True
                continue
            value = parameters[arg].default
            if value == inspect.Parameter.empty:
                all_kwargs[arg] = None
            else:
                all_kwargs[arg] = value

    _log.debug("All kwargs are %s", all_kwargs)
    return all_kwargs, has_kvar


def build_new_mixed_class(
    base_klass: t.Type[T], mixins: t.Sequence[t.Type[M]]
) -> t.Type[t.Union[T, M]]:
    """Build a new mixed class.

    Parameters
    ----------
    base_klass:
        Base class to mix with

    mixins:
        Sequence of mixin classes


    """
    if not hasattr(mixins, "__len__"):
        mixins = [mixins]

    all_classes = tuple(mixins) + tuple([base_klass])
    new_name = "+".join([x.__name__[:10] for x in all_classes])

    new_klass = type(new_name, all_classes, {"__init__": mixed_init})

    return new_klass


def enhance_class(
    base_klass: t.Type[T],
    mixins: t.Sequence[t.Type[M]],
    **kwargs: t.Dict[str, t.Any],
) -> t.Union[T, M]:
    """Build and initialise a new enhanced class.

    Parameters
    ----------
    base_klass:
        Base class to mix with

    mixins:
        Sequence of mixin classes

    kwargs:
        Keyword arguments to pass to the constructor

    Returns
    -------
    t.Union[T, M]:
        A new class that is a subclass of base_klass and all mixins

    Raises
    ------
    KeyError:
        If a keyword argument is not available in the new class

    """
    new_klass = build_new_mixed_class(base_klass, mixins)
    all_kwargs, has_kwarg = determine_mixin_args(new_klass.__bases__)

    for k in kwargs:
        if k not in all_kwargs and not has_kwarg:
            _log.error("Object %s does not have " "parameter %s", new_klass, k)
            _log.error("Available parameters are %s", all_kwargs)
            raise KeyError(f"Object {new_klass} does not have parameter {k}")
    return new_klass(**kwargs)


_mixin_map = {
    TemperatureProfile: TemperatureMixin,
    PressureProfile: PressureMixin,
    Planet: PlanetMixin,
    Star: StarMixin,
    Contribution: ContributionMixin,
    Chemistry: ChemistryMixin,
    ForwardModel: ForwardModelMixin,
    BaseSpectrum: SpectrumMixin,
    Optimizer: OptimizerMixin,
    Instrument: InstrumentMixin,
    Gas: GasMixin,
}
"""Map of base classes to mixin classes."""


def find_mapped_mixin(
    klass: t.Type[T],
) -> t.Type[Mixin]:
    """Find a mapped mixin class.

    Parameters
    ----------
    klass:
        Class to find the map.


    """
    for k in _mixin_map.keys():
        if issubclass(klass, k):
            return _mixin_map[k]

    raise ValueError(f"Class {klass} not found in mixin map")

import typing as t
from functools import partial

from taurex.chemistry import Chemistry
from taurex.contributions import Contribution
from taurex.core.priors import Prior
from taurex.instruments import Instrument
from taurex.log import setup_log
from taurex.model import ForwardModel
from taurex.optimizer import Optimizer
from taurex.planet import Planet
from taurex.pressure import PressureProfile
from taurex.spectrum import BaseSpectrum
from taurex.stellar import Star
from taurex.temperature import TemperatureProfile
from taurex.types import PathLike

from .classfactory import ClassFactory, DiscoverMethod, InputKeywordMethod

log = setup_log(__name__)

T = t.TypeVar("T")

InputT = t.TypeVar("InputT", bound=InputKeywordMethod)
DiscoverT = t.TypeVar("DiscoverT", bound=DiscoverMethod)


class ConfigType(t.TypedDict):
    """Input section."""

    type: t.Optional[str]
    profile_type: t.Optional[str]
    python_file: t.Optional[str]
    ...


def get_keywordarg_dict(klass: t.Type[T]) -> t.Tuple[t.Dict[str, t.Any], bool]:
    """Get all keyword arguments for a class."""
    import inspect

    from taurex.mixin.core import Mixin

    is_mixin = issubclass(klass, Mixin)
    has_kvar = False
    init_dicts = {}
    if not is_mixin:
        init_dicts = {}
        signature = inspect.signature(klass.__init__)

        parameters = signature.parameters

        args = list(parameters.keys())

        if "self" in args:
            args.remove("self")

        for arg in args:
            if parameters[arg].kind == inspect.Parameter.VAR_KEYWORD:
                has_kvar = True
                continue
            if parameters[arg].default == inspect.Parameter.empty:
                init_dicts[arg] = None
            else:
                init_dicts[arg] = parameters[arg].default
    else:
        from taurex.mixin.core import determine_mixin_args

        init_dicts, has_kvar = determine_mixin_args(klass.__bases__)

    log.debug("Init dicts are %s", init_dicts)
    log.debug("Has kvar %s", has_kvar)
    return init_dicts, has_kvar


def create_klass(config: ConfigType, klass: t.Type[T]) -> T:
    """Create a class from a dictionary."""
    kwargs, has_kvar = get_keywordarg_dict(klass)

    for key in config:
        if key in kwargs or has_kvar:
            value = config[key]
            kwargs[key] = value
        else:
            log.error(f"Object {klass.__name__} does not have parameter {key}")
            log.error("Available parameters are %s", kwargs.keys())
            raise KeyError(f"Object {klass.__name__} does not have parameter {key}")
    obj = klass(**kwargs)
    return obj


def generic_factory(profile_type: str, baseclass: t.Type[InputT]) -> t.Type[InputT]:
    """Get a class from input_keyword string."""
    cf = ClassFactory()
    return cf.find_klass_from_keyword(profile_type, baseclass)


# def create_profile(
#     config: ConfigType,
#     baseclass: t.Type[T],
#     alt_type: t.Optional[t.Union[str, t.Sequence[str]]] = None,
#     default_type: t.Optional[str] = None,
# ) -> T:
#     """Create a profile from a dictionary."""
#     config, klass, mixin = determine_klass(
#         config,
#         lambda x: generic_factory(x, baseclass),
#         baseclass=baseclass,
#         alt_type=alt_type,
#         default_type=default_type,
#     )


#     return obj


def create_chemistry(config: ConfigType) -> Chemistry:
    """Chemistry as a special case."""
    from taurex.chemistry import TaurexChemistry
    from taurex.data.profiles.chemistry.gas.gas import Gas

    gases = []
    gas_configs = {k: v for k, v in config.items() if isinstance(v, dict)}

    gases = [
        create_generic(
            {**v, "molecule_name": k},
            Gas,
            alt_type="gas_type",
        )
        for k, v in gas_configs.items()
    ]

    for k in gas_configs.keys():
        config.pop(k)

    log.debug(f"leftover keys {config}")

    obj = create_generic(config, Chemistry, alt_type="chemistry_type")

    if hasattr(obj, "addGas"):
        obj = t.cast(TaurexChemistry, obj)
        for g in gases:
            obj.addGas(g)

    return obj


def determine_klass(
    config: ConfigType,
    baseclass: t.Type[InputT],
    default_type: t.Optional[str] = None,
    alt_type: t.Optional[t.Union[str, t.Sequence[str]]] = None,
) -> t.Tuple[ConfigType, t.Type[InputT], bool]:
    """Determine class from input."""
    from taurex.mixin.core import build_new_mixed_class, find_mapped_mixin

    keyword_types = ["type", "profile_type"]
    if alt_type is not None:
        alt_type = [alt_type] if isinstance(alt_type, str) else alt_type
        keyword_types = keyword_types + alt_type

    check_no_type = len([k for k in keyword_types if k in config]) == 0

    if check_no_type and default_type is not None:
        config["type"] = default_type

    # Check if multiple keyword types exist
    # if they do then we should flag this as an error.

    avail_types = [k for k in keyword_types if k in config]

    if len(avail_types) > 1:
        log.error("Multiple type identifiers found in %s", config)
        raise ValueError("Multiple type identifiers found in %s", config)

    if len(avail_types) == 0:
        log.error("No type identifier found in %s", config)
        raise ValueError(
            f"No keyword types found in {config}, please "
            "include at least 'type' or "
            "'profile_type' in input",
        )

    field = avail_types[0]

    klass_field: t.Union[str, t.Sequence[str]] = config.pop(field)

    klass = None
    is_mixin = False
    if klass_field == "custom":
        try:
            python_file = config.pop("python_file")
        except KeyError as e:
            log.error("No python file for custom profile/model")
            raise KeyError from e

        return config, detect_and_return_klass(python_file, baseclass), False

    split = klass_field.split("+") if isinstance(klass_field, str) else klass_field
    if len(split) == 1:
        klass = generic_factory(klass_field, baseclass)
    else:
        is_mixin = True
        main_klass = generic_factory(split[-1], baseclass)

        mixin_class = find_mapped_mixin(main_klass)

        mixins = [generic_factory(s, mixin_class) for s in split[:-1]]
        klass = build_new_mixed_class(main_klass, mixins)

    return config, klass, is_mixin


def generate_contributions(config: ConfigType) -> t.List[Contribution]:
    """Generate contributions from input."""
    cf = ClassFactory()

    contribs_key = {k: v for k, v in config.items() if isinstance(v, dict)}

    contributions = []

    for key, value in contribs_key.items():
        klass = cf.find_klass_from_keyword(key, Contribution)
        contributions.append(create_klass(value, klass))

    return contributions


def create_generic(
    config: ConfigType,
    baseclass: t.Type[InputT],
    default_type: t.Optional[str] = None,
    alt_type: t.Optional[t.Union[str, t.Sequence[str]]] = None,
) -> InputT:
    """Create a generic class from input."""
    config, klass, _ = determine_klass(
        config,
        baseclass,
        default_type=default_type,
        alt_type=alt_type,
    )

    obj = create_klass(config, klass)

    return obj


def create_prior(prior: str) -> Prior:
    from taurex.util.fitting import parse_priors

    prior_name, args = parse_priors(prior)
    cf = ClassFactory()
    for p in cf.priorKlasses:
        if prior_name in (
            p.__name__,
            p.__name__.lower(),
            p.__name__.upper(),
        ):
            return p(**args)
    else:
        raise ValueError("Unknown Prior Type in input file")


def create_model(
    config: ConfigType,
    gas: Chemistry,
    temperature: TemperatureProfile,
    pressure: PressureProfile,
    planet: Planet,
    star: Star,
    observation: t.Optional[BaseSpectrum] = None,
) -> ForwardModel:
    """Create a forward model from input."""
    log.debug(config)
    config, klass, is_mixin = determine_klass(
        config, ForwardModel, alt_type="model_type"
    )

    log.debug(f"Chosen_model is {klass}")
    kwargs, has_kvar = get_keywordarg_dict(klass)
    log.debug(f"Model kwargs {kwargs}")
    log.debug(f"---------------{gas} {gas.activeGases}--------------")
    if "planet" in kwargs:
        kwargs["planet"] = planet
    if "star" in kwargs:
        kwargs["star"] = star
    if "chemistry" in kwargs:
        kwargs["chemistry"] = gas
    if "temperature_profile" in kwargs:
        kwargs["temperature_profile"] = temperature
    if "pressure_profile" in kwargs:
        kwargs["pressure_profile"] = pressure
    if "observation" in kwargs:
        kwargs["observation"] = observation
    log.debug(f"New Model kwargs {kwargs}")
    log.debug("Creating model---------------")

    kwargs.update({k: v for k, v in config.items() if not isinstance(v, dict)})
    obj = klass(**kwargs)

    contribs = generate_contributions(config)

    for c in contribs:
        obj.add_contribution(c)

    return obj


def detect_and_return_klass(
    python_file: PathLike, baseclass: t.Type[InputT]
) -> t.Type[InputT]:
    """Detect and return class from python file."""
    import importlib.util
    import inspect
    import pathlib

    python_file = pathlib.Path(python_file)

    if not python_file.exists():
        log.error("File %s does not exist", python_file)
        raise FileNotFoundError(f"File {python_file} does not exist")

    spec = importlib.util.spec_from_file_location("foo", python_file)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    classes = [
        m[1]
        for m in inspect.getmembers(foo, inspect.isclass)
        if m[1] is not baseclass and issubclass(m[1], baseclass)
    ]

    if len(classes) == 0:
        log.error("Could not find class of type %s in file %s", baseclass, python_file)
        raise Exception(f"No class inheriting from {baseclass} in " f"{python_file}")
    return classes[0]


# Wrap creation functions for convenience.
create_planet: t.Callable[[ConfigType], Planet] = partial(
    create_generic, baseclass=Planet, alt_type="planet_type", default_type="simple"
)
create_optimizer: t.Callable[[ConfigType], Optimizer] = partial(
    create_generic, baseclass=Optimizer, alt_type="optimizer"
)
create_instrument: t.Callable[[ConfigType], Instrument] = partial(
    create_generic, baseclass=Instrument, alt_type="instrument"
)
create_star: t.Callable[[ConfigType], Star] = partial(
    create_generic, baseclass=Star, alt_type="star_type"
)
create_temperature_profile: t.Callable[[ConfigType], TemperatureProfile] = partial(
    create_generic, baseclass=TemperatureProfile
)
create_pressure_profile: t.Callable[[ConfigType], PressureProfile] = partial(
    create_generic, baseclass=PressureProfile
)
create_spectrum: t.Callable[[ConfigType], BaseSpectrum] = partial(
    create_generic, baseclass=BaseSpectrum, alt_type="observation"
)
create_observation: t.Callable[[ConfigType], BaseSpectrum] = partial(
    create_generic, baseclass=BaseSpectrum, alt_type="observation"
)

"""Unit related utility functions."""

import inspect
from functools import wraps
from typing import Callable
from typing import Optional
from typing import ParamSpec
from typing import TypedDict
from typing import TypeVar
from typing import Union

from astropy import units as u

from .util import convert_to_unit_value


Param = ParamSpec("Param")
RetType = TypeVar("RetType")


# using same as conver_to_unit_value
class UnitArgs(TypedDict):
    """Arguments for unit definitions."""

    target_unit: Union[str, u.Unit]
    default_unit: Union[u.Unit, str]
    equivalencies: Optional[u.Equivalency]


def validate_arg_units(
    unit_def: dict[str, UnitArgs],
) -> Callable[[Callable[Param, RetType]], Callable[Param, RetType]]:
    """Convert supplied function arguments to the specified units.

    Parameters
    ----------
    unit_def
        Map parameter names to unit definitions. A definition for ``*args``
        applies to each element, and one for ``**kwargs`` applies to each value.
        Definitions for individual keyword names override the ``**kwargs``
        definition. Omitted arguments retain their function defaults unchanged.

    Returns
    -------
    Callable
        Decorator that converts arguments before calling the function.
    """

    def convert(value, definition):
        if definition is None:
            return value
        return convert_to_unit_value(
            value,
            definition["target_unit"],
            definition.get("default_unit"),
            definition.get("equivalencies"),
        )

    def decorator(func: Callable[Param, RetType]) -> Callable[Param, RetType]:
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args: Param.args, **kwargs: Param.kwargs) -> RetType:
            bound = sig.bind(*args, **kwargs)
            for name, value in bound.arguments.items():
                definition = unit_def.get(name)
                kind = sig.parameters[name].kind
                if kind == inspect.Parameter.VAR_POSITIONAL:
                    value = tuple(convert(item, definition) for item in value)
                elif kind == inspect.Parameter.VAR_KEYWORD:
                    value = {
                        key: convert(item, unit_def.get(key, definition))
                        for key, item in value.items()
                    }
                else:
                    value = convert(value, definition)
                bound.arguments[name] = value

            return func(*bound.args, **bound.kwargs)

        return wrapper

    return decorator


# Some default TauREX unit constants

DEFAULT_TEMPERATURE: UnitArgs = {
    "default_unit": u.K,
    "target_unit": u.K,
    "equivalencies": u.temperature(),
}

DEFAULT_PRESSURE: UnitArgs = {
    "default_unit": u.Pa,
    "target_unit": u.Pa,
}

DEFAULT_SPECTRUM: UnitArgs = {
    "default_unit": u.k,
    "target_unit": u.k,
    "equivalencies": u.spectral(),
}

"""Unit related utility functions."""

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


# Need a decorator of decorator to add args
# Does not support var_args or var_kwargs, neet to test methods
# make sure self doesnt break this.
def validate_arg_units(
    unit_def: dict[str, UnitArgs]
) -> Callable[[Callable[Param, RetType]], Callable[Param, RetType]]:
    """Evaluate emission flux and integrate quadratures for flux.

    Parameters
    ----------
    unit_def
        Define units in function arguments / kwargs

    Returns
    -------
    decorator for function.

    """
    import inspect

    def decorator(func: Callable[Param, RetType]) -> Callable[Param, RetType]:
        def wrapper(*args: Param.args, **kwargs: Param.kwargs) -> RetType:
            new_args = []
            new_kwargs = {}
            # Handle args
            sig = inspect.signature(func)
            for p, value in zip(sig.parameters, args, strict=False):

                if p in unit_def:
                    value = convert_to_unit_value(
                        value,
                        unit_def[p]["target_unit"],
                        unit_def[p].get("default_unit"),
                        unit_def[p].get("equivalencies"),
                    )

                new_args.append(value)

            # Handle kwargs
            for k, v in kwargs.items():
                if k in unit_def:
                    v = convert_to_unit_value(
                        v,
                        unit_def[k]["target_unit"],
                        unit_def[k].get("default_unit"),
                        unit_def[k].get("equivalencies"),
                    )

                new_kwargs[k] = v

            return func(*new_args, **new_kwargs)

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

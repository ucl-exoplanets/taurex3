"""Module to safely call attributes and functions across modules."""

import typing as t


def getattr_recursive(obj: t.Any, attr: str) -> t.Any:
    """Get attribute recursively."""
    split = attr.split(".")
    if len(split) == 1:
        return getattr(obj, split[0])
    else:
        return getattr_recursive(getattr(obj, split[0]), ".".join(split[1:]))


def setattr_recursive(obj: t.Any, attr: str, value: t.Any) -> None:
    """Set attribute recursively."""
    split = attr.split(".")
    if len(split) == 1:
        setattr(obj, split[0], value)
    else:
        setattr_recursive(getattr(obj, split[0]), ".".join(split[1:]), value)


def runfunc_recursive(obj: t.Any, func: str, *args: t.Any, **kwargs: t.Any) -> t.Any:
    """Run function recursively."""
    split = func.split(".")
    if len(split) == 1:
        return getattr(obj, split[0])(*args, **kwargs)
    else:
        return runfunc_recursive(
            getattr(obj, split[0]), ".".join(split[1:]), *args, **kwargs
        )

"""Core classes for Taurex."""
from taurex.data.citation import Citable, to_bibtex, unique_citations_only
from taurex.data.fittable import (
    DerivedType,
    Fittable,
    FittingType,
    derivedparam,
    fitparam,
)
from taurex.output.output import Output

"""Just contains a singleton class. Pretty useful"""


class Singleton:
    """
    A singleton for your usage. When inheriting do not implement __init__ instead
    override :func:`init`


    """

    def __new__(cls, *args, **kwds):
        it = cls.__dict__.get("__it__")
        if it is not None:
            return it
        cls.__it__ = it = object.__new__(cls)
        it.init(*args, **kwds)
        return it

    def init(self, *args, **kwds):
        """Override to act as an init"""
        pass


__all__ = [
    "fitparam",
    "derivedparam",
    "Fittable",
    "Citable",
    "unique_citations_only",
    "to_bibtex",
    "Output",
    "Singleton",
    "FittingType",
    "DerivedType",
]

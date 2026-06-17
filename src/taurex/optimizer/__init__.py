"""Optimizer module."""

from .nestle import NestleOptimizer
from .optimizer import Optimizer


try:
    from .multinest import MultiNestOptimizer
except ImportError:
    pass

try:
    from .polychord import PolyChordOptimizer
except ImportError:
    pass

try:
    from .dypolychord import DyPolyChordOptimizer
except ImportError:
    pass

__all__ = [
    "NestleOptimizer",
    "Optimizer",
    "MultiNestOptimizer",
    "PolyChordOptimizer",
    "DyPolyChordOptimizer",
]

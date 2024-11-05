"""Modules relating to defining stellar properties of the model."""

from .phoenix import PhoenixStar
from .star import BlackbodyStar

__all__ = ["BlackbodyStar", "PhoenixStar"]

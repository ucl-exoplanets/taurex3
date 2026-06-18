"""Modules relating to defining stellar properties of the model."""

from .phoenix import PhoenixStar
from .star import BlackbodyStar
from .star import Star


__all__ = ["BlackbodyStar", "PhoenixStar", "Star"]

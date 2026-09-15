"""Modules relating to defining stellar properties of the model."""

from .phoenix import PhoenixStar
from .phoenix4all import Phoenix4AllStar
from .star import BlackbodyStar
from .star import Star


__all__ = ["BlackbodyStar", "PhoenixStar", "Phoenix4AllStar", "Star"]

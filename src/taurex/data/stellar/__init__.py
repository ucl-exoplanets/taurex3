"""Modules relating to defining stellar properties of the model."""

from .phoenix import PhoenixStar
from .phoenix4all import Phoenix4AllStar
from .star import BlackbodyStar

__all__ = ["BlackbodyStar", "PhoenixStar", "Phoenix4AllStar"]

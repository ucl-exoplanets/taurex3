"""Writeable class."""
from abc import ABCMeta, abstractmethod


class Writeable(metaclass=ABCMeta):
    """Abstract class for writeable objects."""

    @abstractmethod
    def write(self, output):
        pass

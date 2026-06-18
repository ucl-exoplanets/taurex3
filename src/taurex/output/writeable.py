"""Writeable class."""

from abc import ABCMeta
from abc import abstractmethod


class Writeable(metaclass=ABCMeta):
    """Abstract class for writeable objects."""

    @abstractmethod
    def write(self, output):
        """Write to output.

        Parameters
        ----------
        output : OutputGroup
            Output group to write to.

        """

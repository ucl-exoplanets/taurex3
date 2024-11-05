"""An object that globally caches variables."""

import typing as t

from taurex.log import Logger

from .singleton import Singleton


class GlobalCache(Singleton):
    """Allows for the storage of global variables."""

    def init(self):
        self.variable_dict: t.Dict[str, t.Any] = {}
        self.log = Logger("GlobalCache")

    def __getitem__(self, key: str) -> t.Any:
        """Get a variable from the cache."""
        return self.variable_dict.get(key, None)

    def __setitem__(self, key: str, value: t.Any) -> t.Any:
        """Set a variable in the cache.

        Parameters
        ----------
        key : str
            The key to store the variable under.

        value : Any
            Value to store.

        """
        self.variable_dict[key] = value

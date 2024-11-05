"""Output base class"""

import typing as t

import numpy as np
import numpy.typing as npt

from taurex.log import Logger
from taurex.types import AnyValType, ArrayType, ScalarType

MetadataType = t.Dict[str, AnyValType]
"""Type for metadata"""


class OutputGroup(Logger):
    """Stores output data in a hierarchical structure."""

    def __init__(self, name: str):
        super().__init__(name)
        self._name = name

    def write_array(
        self,
        array_name: str,
        array: ArrayType,
        metadata: t.Optional[MetadataType] = None,
    ) -> None:
        raise NotImplementedError

    def write_list(
        self,
        list_name: str,
        list_array: npt.ArrayLike,
        metadata: t.Optional[MetadataType] = None,
    ) -> None:
        arr = np.array(list_array)
        self.write_array(list_name, arr)

    def write_scalar(
        self,
        scalar_name: str,
        scalar: ScalarType,
        metadata: t.Optional[MetadataType] = None,
    ) -> None:
        raise NotImplementedError

    def write_string(
        self, string_name: str, string: str, metadata: t.Optional[MetadataType] = None
    ) -> None:
        raise NotImplementedError

    def write_string_array(
        self,
        string_name: str,
        string_array: t.Optional[str],
        metadata: t.Optional[MetadataType] = None,
    ) -> None:
        raise NotImplementedError

    def create_group(self, group_name: str) -> "OutputGroup":
        """Create a group."""
        raise NotImplementedError

    def store_dictionary(
        self, dictionary: t.Dict[str, t.Any], group_name: t.Optional[str] = None
    ) -> None:
        """Store a dictionary in the output."""
        from taurex.util import recursively_save_dict_contents_to_output

        group = self
        if group_name is not None:
            group = self.create_group(group_name)
        recursively_save_dict_contents_to_output(group, dictionary)


class Output(OutputGroup):
    """Base calss for handling outputs from Taurex3"""

    def __init__(self, name: str) -> None:
        """Initialize output.

        Parameters
        ----------
        name : str
            Name for logging purposes.

        """
        super().__init__(name)

    def open(self) -> None:
        """Open output."""
        raise NotImplementedError

    def close(self) -> None:
        """Close output."""
        raise NotImplementedError

    def __enter__(self) -> None:
        """Enter context manager."""
        self.open()
        return self

    def __exit__(self, type: t.Any, value: t.Any, traceback: t.Any) -> None:
        """Exit context manager."""
        self.close()

    def store_dictionary(
        self, dictionary: t.Dict[str, t.Any], group_name: t.Optional[str] = None
    ) -> None:
        """Store a dictionary in the output."""
        from taurex.util import recursively_save_dict_contents_to_output

        out = self
        if group_name is not None:
            out = self.create_group(group_name)

        recursively_save_dict_contents_to_output(out, dictionary)

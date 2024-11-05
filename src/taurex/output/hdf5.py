"""Outputs using HDF5 format."""

import datetime
import typing as t

import h5py
import numpy as np

from taurex.mpi import only_master_rank
from taurex.types import ArrayType, ScalarType

from .output import MetadataType, Output, OutputGroup


class HDF5OutputGroup(OutputGroup):
    """Stores output data in the HDF5 hierarchical structure."""

    def __init__(
        self, entry: t.Optional[h5py.Group] = None, name: t.Optional[str] = None
    ) -> None:
        """Initialize HDF5 output group."""
        super().__init__(name=name or "HDF5Group")
        self._entry = entry

    @only_master_rank
    def write_array(
        self,
        array_name: str,
        array: ArrayType,
        metadata: t.Optional[MetadataType] = None,
    ) -> None:
        array = np.array(array)
        ds = self._entry.create_dataset(
            str(array_name), data=array, shape=array.shape, dtype=array.dtype
        )
        if metadata:
            for k, v in metadata.items():
                ds.attrs[k] = v

    @only_master_rank
    def write_scalar(
        self,
        scalar_name: str,
        scalar: ScalarType,
        metadata: t.Optional[MetadataType] = None,
    ) -> None:
        ds = self._entry.create_dataset(str(scalar_name), data=scalar)
        if metadata:
            for k, v in metadata.items():
                ds.attrs[k] = v

    @only_master_rank
    def write_string(
        self, string_name: str, string: str, metadata: t.Optional[MetadataType] = None
    ) -> None:
        ds = self._entry.create_dataset(str(string_name), data=string)
        if metadata:
            for k, v in metadata.items():
                ds.attrs[k] = v

    def create_group(self, group_name: str) -> "HDF5OutputGroup":
        """Create a group."""
        entry = None
        if self._entry is not None:
            entry = self._entry.create_group(str(group_name))
            return HDF5OutputGroup(entry=entry, name=group_name)
        else:
            return HDF5OutputGroup(name=group_name, entry=None)

    @only_master_rank
    def write_string_array(
        self,
        string_name: str,
        string_array: t.Sequence[str],
        metadata: t.Dict[str, t.Any] = None,
    ) -> None:
        """Write a string array to the output."""
        ascii_lst = [n.encode("ascii", "ignore") for n in string_array]
        ds = self._entry.create_dataset(
            str(string_name), (len(ascii_lst), 1), "S64", ascii_lst
        )

        if metadata:
            for k, v in metadata.items():
                ds.attrs[k] = v


class HDF5Output(Output, HDF5OutputGroup):
    """Output using HDF5 format."""

    def __init__(self, filename, append=False):
        Output.__init__(self, "HDF5Output")
        HDF5OutputGroup.__init__(self, name="HDF5Output", entry=None)

        self.filename = filename
        self._append = append
        self.fd: h5py.File = None

    @only_master_rank
    def open(self) -> None:
        """Open file."""
        self.fd = self._open_file(self.filename)
        self._entry = self.fd

    def _open_file(self, fname: str) -> h5py.File:
        """Open file and write header."""
        from taurex._version import __version__

        mode = "w"
        if self._append:
            mode = "a"

        fd = h5py.File(fname, mode=mode)
        fd.attrs["file_name"] = fname
        fd.attrs["file_time"] = datetime.datetime.now().isoformat()
        fd.attrs["creator"] = self.__class__.__name__
        fd.attrs["HDF5_Version"] = h5py.version.hdf5_version
        fd.attrs["h5py_version"] = h5py.version.version
        fd.attrs["program_name"] = "TauREx"
        fd.attrs["program_version"] = __version__
        return fd

    def close(self) -> None:
        """Close file."""
        if self.fd:
            self.fd.flush()
            self.fd.close()
        self.fd = None

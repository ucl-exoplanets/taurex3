import typing as t

import numpy as np
import numpy.typing as npt

from taurex.types import PathLike
from taurex.util import wnwidth_to_wlwidth

from .array import ArraySpectrum


class TaurexSpectrum(ArraySpectrum):
    """Observation is a taurex spectrum from a HDF5 file.

    An instrument function must have been used for this to work

    """

    def __init__(self, filename: t.Optional[PathLike] = None):
        """Initialize and load from HDF5 file.

        Parameters
        ----------
        filename:
            Filename of HDF5 file

        Raises
        ------
        KeyError
            If instrument output not found in HDF5 file

        """
        super().__init__(self._load_from_hdf5(filename))

    def _load_from_hdf5(self, filename: PathLike) -> npt.NDArray[np.float64]:
        """Load from HDF5 file.

        Parameters
        ----------
        filename:
            Filename of HDF5 file

        Returns
        -------
        np.ndarray[np.float64]
            Array of shape (nbins, 4) with columns:
            1. wavelength (um)
            2. spectral data
            3. error
            4. bin width (um)

        Raises
        ------
        KeyError
            If instrument output not found in HDF5 file

        """
        import h5py

        with h5py.File(filename, "r") as f:
            try:
                wngrid = f["Output"]["Spectra"]["instrument_wngrid"][:]
            except KeyError as e:
                self.error(
                    "Could not find instrument outputs in HDF5, "
                    "this was caused either by the HDF5 being a "
                    "retrieval output or not running with some "
                    "form of instrument in the forward model"
                    " input par file"
                )
                raise KeyError("Instrument output not found") from e

            spectrum = f["Output"]["Spectra"]["instrument_spectrum"][:]
            noise = f["Output"]["Spectra"]["instrument_noise"][:]
            wnwidth = f["Output"]["Spectra"]["instrument_wnwidth"][:]

        wlgrid = 10000 / wngrid

        wlwidth = wnwidth_to_wlwidth(wngrid, wnwidth)

        return np.vstack((wlgrid, spectrum, noise, wlwidth)).T

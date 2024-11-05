"""Module for the base binning class"""

import typing as t

import numpy as np
import numpy.typing as npt

from taurex import OutputSize
from taurex.log import Loggable
from taurex.types import ModelOutputType
from taurex.util import compute_bin_edges

BinDownType = t.Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    t.Optional[npt.NDArray[np.float64]],
    t.Optional[npt.NDArray[np.float64]],
]


class BinnedSpectrumType(t.TypedDict):
    """Binned spectrum type."""

    native_wngrid: npt.NDArray[np.float64]
    native_wlgrid: npt.NDArray[np.float64]
    binned_wngrid: t.Optional[npt.NDArray[np.float64]]
    binned_wlgrid: t.Optional[npt.NDArray[np.float64]]
    native_spectrum: npt.NDArray[np.float64]
    binned_spectrum: npt.NDArray[np.float64]
    native_wnwidth: npt.NDArray[np.float64]
    native_wlwidth: npt.NDArray[np.float64]
    binned_wnwidth: t.Optional[npt.NDArray[np.float64]]
    binned_wlwidth: t.Optional[npt.NDArray[np.float64]]
    binned_tau: t.Optional[npt.NDArray[np.float64]]
    native_tau: t.Optional[npt.NDArray[np.float64]]


class Binner(Loggable):
    """
    *Abstract class*

    The binner class deals with binning down spectra to different resolutions.
    It also provides a method to generate spectrum output format from a forward
    model result in the form of a dictionary.
    Using this class does not need to be restricted to TauREx3 results and can
    be used to bin down any arbitrary spectra.
    """

    def __init__(self):
        super().__init__()

    def bindown(
        self,
        wngrid: npt.NDArray[np.float64],
        spectrum: npt.NDArray[np.float64],
        grid_width: t.Optional[npt.NDArray[np.float64]] = None,
        error: t.Optional[npt.NDArray[np.float64]] = None,
    ) -> BinDownType:
        """Bin down a spectrum from a high resolution to a lower resolution.
        **Requires implementation**

        This should handle the binning of a spectrum passed into the function.
        Parameters given are guidelines on expectation of usage.

        Parameters
        ----------
        wngrid : :obj:`array`
            The wavenumber grid of the spectrum to be binned down.
            Generally the 'native' wavenumber grid

        spectrum: :obj:`array`
            The spectra we wish to bin-down. Must be same shape as
            ``wngrid``.

        grid_width: :obj:`array`, optional
            Wavenumber grid full-widths for the spectrum to be binned down.
            Must be same shape as ``wngrid``.
            Optional, generally if you require this but the user does not pass
            it then you must compute it yourself using ``wngrid``. This can
            be done easily using the function
            func:`~taurex.util.compute_bin_edges`.

        error: :obj:`array`, optional
            Associated errors or noise of the spectrum. Must be same shape
            as ``wngrid``.Optional parameter, when implementing you must
            deal with the cases where either the error is passed or not passed.

        Returns
        -------
        binned_wngrid : :obj:`array`
            New wavenumber grid

        spectrum: :obj:`array`
            Binned spectrum.

        grid_width: :obj:`array`
            New grid-widths

        error: :obj:`array` or None
            If passed, should be the binned error otherwise None

        Raises
        ------
        NotImplementedError
            If not implemented

        """
        raise NotImplementedError

    def bin_model(self, model_output: ModelOutputType) -> BinDownType:
        """
        Bins down a TauREx3 forward model.
        This automatically splits the output and passes it to
        the :func:`bindown` function.
        Its general usage is of the form:

        >>> fm = TransmissionModel()
        >>> fm.build()
        >>> result = fm.model()
        >>> binner.bin_model(result)

        Or in a single line:

        >>> binner.bin_model(fm.model())


        Parameters
        ----------
        model_output: obj:`tuple`
            Result from running a TauREx3 forward model

        Returns
        -------
        See :func:`bindown`


        """
        return self.bindown(model_output[0], model_output[1])

    def generate_spectrum_output(
        self,
        model_output: ModelOutputType,
        output_size: t.Optional[OutputSize] = OutputSize.heavy,
    ) -> BinnedSpectrumType:
        """Generate binned spectrum output.

        Given a forward model output, generate a dictionary
        that can be used to store to file. This can include
        storing the native and binned spectrum.
        Not necessary for the function of the class but useful for
        full intergation into TauREx3, especially when storing results
        from a retrieval.
        Can be overwritten to store more information.

        Parameters
        ----------
        model_output: obj:`tuple`
            Result from running a TauREx3 forward model

        output_size: :class:`~taurex.taurexdefs.OutputSize`
            Size of the output.


        Returns
        -------
        :obj:`dict`:
            Dictionary of spectra


        """
        output: BinnedSpectrumType = {}

        wngrid, flux, tau, extra = model_output

        output["native_wngrid"] = wngrid
        output["native_wlgrid"] = 10000 / wngrid
        output["native_spectrum"] = flux
        output["binned_spectrum"] = self.bindown(wngrid, flux)[1]
        output["native_wnwidth"] = compute_bin_edges(wngrid)[-1]
        output["native_wlwidth"] = compute_bin_edges(10000 / wngrid)[-1]
        if output_size > OutputSize.lighter:
            output["binned_tau"] = self.bindown(wngrid, tau)[1]
            if output_size > OutputSize.light:
                output["native_tau"] = tau

        return output

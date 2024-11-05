import typing as t

import numpy as np
import numpy.typing as npt

from taurex import OutputSize
from taurex.binning import BinDownType, BinnedSpectrumType, Binner
from taurex.types import ModelOutputType


class NativeBinner(Binner):
    """A `do-nothing` binner.

    This is useful when the pipeline expects
    a binner but none is given. Simplifies implementation and also
    handles dictionary writing of the forward model.
    """

    def bindown(
        self,
        wngrid: npt.NDArray[np.float64],
        spectrum: npt.NDArray[np.float64],
        grid_width: t.Optional[npt.NDArray[np.float64]] = None,
        error: t.Optional[npt.NDArray[np.float64]] = None,
    ) -> BinDownType:
        """Bin down a spectrum from a high resolution to a lower resolution.

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

        """
        return wngrid, spectrum, grid_width, error

    def bin_model(self, model_output: ModelOutputType) -> BinDownType:
        """Does nothing, only returns function arguments"""
        return *model_output[:2], None, None

    def generate_spectrum_output(
        self,
        model_output: ModelOutputType,
        output_size: t.Optional[OutputSize] = OutputSize.heavy,
    ) -> BinnedSpectrumType:
        """Generate spectrum output for lightcurves.

        Parameters
        ----------

        model_output: obj:`tuple`
            Result from running a TauREx3 forward model

        output_size: :class:`~taurex.taurexdefs.OutputSize`
            Size of the output.

        Returns
        -------
        :obj:`dict`:
            Dictionary of spectra containing both lightcurves
            and spectra.


        """
        output = {}

        wngrid, flux, tau, extra = model_output

        output["native_wngrid"] = wngrid
        output["native_wlgrid"] = 10000 / wngrid
        output["native_spectrum"] = flux

        if output_size > OutputSize.light:
            output["native_tau"] = tau

        return output

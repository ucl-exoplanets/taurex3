"""Contains the basic definition of an observed spectrum for TauRex 3."""

import typing as t

import numpy as np
import numpy.typing as npt

from taurex.binning import Binner
from taurex.core import DerivedType, Fittable, FittingType
from taurex.log import Logger
from taurex.output import OutputGroup
from taurex.output.writeable import Writeable


class BaseSpectrum(Logger, Fittable, Writeable):
    """A base class that represents spectrums.

    *Abstract class*

    A base class where spectrums are loaded (or later created). This
    is used to either plot against the forward model or passed into the
    optimizer to be used to fit the forward model.
    """

    def __init__(self, name: str):
        """Initialize.

        Parameters
        ----------

        name : str
            Name to be used in logging

        """
        Logger.__init__(self, name)
        Fittable.__init__(self)

    def create_binner(self) -> Binner:
        """Creates the appropriate binning object."""
        from taurex.binning import FluxBinner

        return FluxBinner(wngrid=self.wavenumberGrid, wngrid_width=self.binWidths)

    @property
    def spectrum(self) -> npt.NDArray[np.float64]:
        """Spectrum of the observation.

        **Requires Implementation**


        Should return the observed spectrum.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    @property
    def rawData(self) -> t.Any:  # noqa: N802
        """Raw data of the observation.
        **Requires Implementation**


        Should return the raw data set.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    @property
    def wavelengthGrid(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Wavelength grid of the spectrum in microns.


        **Requires Implementation**


        Should return the wavelength grid of the spectrum in microns.
        This does not need to necessarily match the shape of :func:`spectrum`

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    @property
    def wavenumberGrid(self):  # noqa: N802
        """Wavenumber grid in :math:`cm^{-1}`

        Returns
        -------
        wngrid : :obj:`array`

        """
        return 10000 / self.wavelengthGrid

    @property
    def binEdges(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Bin edges of the wavenumber grid.
        **Requires Implementation**


        Should return the bin edges of the wavenumber grid

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    @property
    def binWidths(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Widths of each bin in the wavenumber grid
        **Requires Implementation**


        Should return the widths of each bin in the wavenumber grid

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    @property
    def errorBar(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Return error or uncertainty of the spectrum.
        **Requires Implementation**


        Should return the error. *Must* be the same shape as
        :func:`spectrum`

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    @property
    def fittingParameters(self) -> t.Dict[str, FittingType]:  # noqa: N802
        """Return fitting parameters."""
        return self.fitting_parameters()

    @property
    def derivedParameters(self) -> t.Dict[str, DerivedType]:  # noqa: N802
        """Return derived parameters."""
        return self.derived_parameters()

    def write(self, output: OutputGroup) -> OutputGroup:
        """Write spectrum to output group."""
        output.write_array("wlgrid", self.wavelengthGrid)
        output.write_array("spectrum", self.spectrum)
        output.write_array("binedges", self.binEdges)
        output.write_array("binwidths", self.binWidths)
        output.write_array("errorbars", self.errorBar)

        return output

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        raise NotImplementedError

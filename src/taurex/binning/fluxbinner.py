"""Module for the flux binner class."""

import typing as t

import numpy as np
import numpy.typing as npt

from taurex import OutputSize
from taurex.util import compute_bin_edges

from ..types import ModelOutputType
from .binner import BinDownType
from .binner import BinnedSpectrumType
from .binner import Binner


class FluxBinnerOld(Binner):
    """Bins to a wavenumber grid given by ``wngrid`` using a more accurate method.

    This method takes into account the amount of contribution from each
    native bin. This method also handles cases where bins are not
    continuous and/or overlapping.

    Parameters
    ----------
    wngrid: :obj:`array`
        Wavenumber grid

    wngrid_width: :obj:`array`, optional
        Must have same shape as ``wngrid``
        Full bin widths for each wavenumber grid point
        given in ``wngrid``. If not provided then
        this is automatically computed from ``wngrid``.

    """

    def __init__(
        self,
        wngrid: npt.NDArray[np.float64],
        wngrid_width: t.Optional[npt.NDArray[np.float64]] = None,
    ):
        """Initialize FluxBinner.

        Parameters
        ----------
        wngrid : obj:`array`
            Wavenumber grid
        wngrid_width : obj:`array`, optional
            Must have same shape as ``wngrid``
            Full bin widths for each wavenumber grid point
            given in ``wngrid``. If not provided then
            this is automatically computed from ``wngrid``.

        """
        super().__init__()
        sort_grid = wngrid.argsort()
        self._wngrid = wngrid[sort_grid]
        self._wngrid_width = wngrid_width

        if self._wngrid_width is None:
            self._wngrid_width = compute_bin_edges(self._wngrid)[-1]
        elif hasattr(self._wngrid_width, "__len__"):
            if len(self._wngrid_width) != len(self._wngrid):
                raise ValueError(
                    "Wavenumber width should be signel value or "
                    "same shape as wavenumber grid"
                )
            self._wngrid_width = wngrid_width[sort_grid]

        if not hasattr(self._wngrid_width, "__len__"):
            self._wngrid_width = np.ones_like(self._wngrid) * self._wngrid_width

    def bindown(
        self,
        wngrid: npt.NDArray[np.float64],
        spectrum: npt.NDArray[np.float64],
        grid_width: t.Optional[npt.NDArray[np.float64]] = None,
        error: t.Optional[npt.NDArray[np.float64]] = None,
    ) -> BinDownType:
        """Bins down spectrum.

        Parameters
        ----------
        wngrid : :obj:`array`
            The wavenumber grid of the spectrum to be binned down.

        spectrum: :obj:`array`
            The spectra we wish to bin-down. Must be same shape as
            ``wngrid``.

        grid_width: :obj:`array`, optional
            Wavenumber grid full-widths for the spectrum to be binned down.
            Must be same shape as ``wngrid``.
            Optional.

        error: :obj:`array`, optional
            Associated errors or noise of the spectrum. Must be same shape
            as ``wngrid``.Optional parameter.

        Returns
        -------
        binned_wngrid : :obj:`array`
            New wavenumber grid

        spectrum: :obj:`array`
            Binned spectrum.

        grid_width: :obj:`array`
            New grid-widths

        error: :obj:`array` or None
            Binned error if given else ``None``

        """
        sorted_input = wngrid.argsort()
        wngrid = wngrid[sorted_input]
        spectrum = spectrum[..., sorted_input]
        if error is not None:
            error = error[..., sorted_input]

        bin_spectrum = np.zeros(spectrum[..., 0].shape + self._wngrid.shape)

        if error is not None:
            bin_error = np.zeros(spectrum[..., 0].shape + self._wngrid.shape)
        else:
            bin_error = None

        old_spect_wn = wngrid

        old_spect_flux = spectrum
        old_spect_err = error

        old_spect_width = grid_width

        if old_spect_width is None:
            old_spect_width = compute_bin_edges(old_spect_wn)[-1]
        else:
            old_spect_width = old_spect_width[sorted_input]

        old_spect_min = old_spect_wn - old_spect_width / 2
        old_spect_max = old_spect_wn + old_spect_width / 2

        new_spec_lhs = self._wngrid
        new_spec_rhs = self._wngrid_width

        new_spec_wn = self._wngrid

        new_spec_wn_min = new_spec_lhs - new_spec_rhs / 2
        new_spec_wn_max = new_spec_lhs + new_spec_rhs / 2

        save_start = 0
        save_stop = 0

        for idx, res in enumerate(
            zip(new_spec_wn, new_spec_wn_min, new_spec_wn_max, strict=True)
        ):
            wn, wn_min, wn_max = res
            sum_spectrum = 0
            sum_noise = 0
            sum_weight = 0

            save_start = np.searchsorted(old_spect_max, wn_min, side="right")
            save_stop = np.searchsorted(old_spect_min[1:], wn_max, side="right")

            save_stop = min(save_stop, old_spect_min.shape[0] - 1)
            save_start = min(save_start, old_spect_min.shape[0] - 1)

            if (
                not wn_min <= old_spect_max[save_start]
                or not old_spect_min[save_stop] <= wn_max
            ):
                continue

            spect_min = old_spect_min[save_start : save_stop + 1]
            spect_max = old_spect_max[save_start : save_stop + 1]

            weight = (np.minimum(wn_max, spect_max) - np.maximum(spect_min, wn_min)) / (
                wn_max - wn_min
            )

            sum_weight = np.sum(weight)

            sum_spectrum = np.sum(
                weight / sum_weight * old_spect_flux[..., save_start : save_stop + 1],
                axis=-1,
            )

            if error is not None:
                sum_noise = np.sum(
                    weight
                    * weight
                    * old_spect_err[..., save_start : save_stop + 1] ** 2,
                    axis=0,
                )

                sum_noise = np.sqrt(sum_noise / sum_weight / sum_weight)

            bin_spectrum[..., idx] = sum_spectrum

            if error is not None:
                bin_error[idx] = sum_noise

        return self._wngrid, bin_spectrum, bin_error, self._wngrid_width

    def generate_spectrum_output(
        self,
        model_output: ModelOutputType,
        output_size: t.Optional[OutputSize] = OutputSize.heavy,
    ) -> BinnedSpectrumType:
        """Generate spectrum output."""
        output = super().generate_spectrum_output(model_output, output_size=output_size)
        output["binned_wngrid"] = self._wngrid
        output["binned_wlgrid"] = 10000 / self._wngrid
        output["binned_wnwidth"] = self._wngrid_width
        output["binned_wlwidth"] = 1.0 / self._wngrid_width
        return output


class FluxBinner(Binner):
    """Bins to a wavenumber or wavelength grid using a more accurate method.

    This method that takes into account the amount
    of contribution from each native bin. This method also
    handles cases where bins are not continuous and/or
    overlapping, and can handle wavelength grids as well
    as wavenumber grids.
    """

    def __init__(
        self,
        wngrid: t.Optional[npt.NDArray[np.float64]] = None,
        wngrid_width: t.Optional[npt.NDArray[np.float64]] = None,
        wlgrid: t.Optional[npt.NDArray[np.float64]] = None,
        wlgrid_width: t.Optional[npt.NDArray[np.float64]] = None,
    ):
        """Initializes the FluxBinner.

        Parameters
        ----------
        wngrid: :obj:`array`, optional
            Wavenumber grid. If not set, then ``wlgrid`` must be.

        wngrid_width: :obj:`array`, optional
            Must have same shape as ``wngrid``
            Full bin widths for each wavenumber grid point
            given in ``wngrid``. If not provided then
            this is automatically computed from ``wngrid``.

        wlgrid: :obj:`array`, optional
            Wavelength grid. If not set, then ``wngrid`` must be.

        wlgrid_width: :obj:`array`, optional
            Must have same shape as ``wlgrid``
            Full bin widths for each wavelength grid point
            given in ``wlgrid``. If not provided then
            this is automatically computed from ``wlgrid``.
        """
        super().__init__()

        if (wngrid is None and wlgrid is None) or (
            wngrid is not None and wlgrid is not None
        ):
            raise ValueError("You must specify exactly one between wngrid and wlgrid")
        if wngrid is not None and wlgrid_width is not None:
            raise ValueError("You cannot use wlgrid_width with wngrid")
        if wlgrid is not None and wngrid_width is not None:
            raise ValueError("You cannot use wngrid_width with wlgrid")

        if wngrid is not None:
            sort_grid = wngrid.argsort()
            self._bin_centers = wngrid[sort_grid]
            self._bin_widths = wngrid_width
            self._in_wl = False
        else:
            sort_grid = wlgrid.argsort()
            self._bin_centers = wlgrid[sort_grid]
            self._bin_widths = wlgrid_width
            self._in_wl = True

        if self._bin_widths is None:
            self._bin_widths = compute_bin_edges(self._bin_centers)[-1]
        elif hasattr(self._bin_widths, "__len__"):
            if len(self._bin_widths) != len(self._bin_centers):
                raise ValueError(
                    f"w{'nl'[self._in_wl]}grid_width should be a scalar or\
                    np.array with the same shape as w{'nl'[self._in_wl]}grid"
                )
            self._bin_widths = self._bin_widths[sort_grid]
        else:
            self._bin_widths = np.ones_like(self._bin_centers) * self._bin_widths

    def bindown(
        self,
        wngrid: npt.NDArray[np.float64],
        spectrum: npt.NDArray[np.float64],
        grid_width: t.Optional[npt.NDArray[np.float64]] = None,
        error: t.Optional[npt.NDArray[np.float64]] = None,
    ) -> BinDownType:
        """Bins down spectrum.

        Parameters
        ----------
        wngrid : :obj:`array`
            The wavenumber grid of the spectrum to be binned down.

        spectrum: :obj:`array`
            The spectra we wish to bin-down. Must be same shape as
            ``wngrid``. If the FluxBinner was initialized with a wavelength grid,
            it must be a density over wavelength (default for spectra computed
            by taurex). If instead the FluxBinner was initialized with a
            wavenumber grid, it must be a density over wavenumbers.

        grid_width: :obj:`array`, optional
            Wavenumber grid full-widths for the spectrum to be binned down.
            Must be same shape as ``wngrid``.
            Optional.

        error: :obj:`array`, optional
            Associated errors or noise of the spectrum. Must be same shape
            as ``wngrid``.Optional parameter.

        Returns
        -------
        bin_centers: :obj:`array`
            Centers of the new bins

        spectrum: :obj:`array`
            Binned spectrum.

        bin_widths: :obj:`array`
            Width of the bins

        error: :obj:`array` or None
            Binned error if given else ``None``
        """
        # sort inputs
        sorted_input = wngrid.argsort()
        old_spect_wn = wngrid[sorted_input]
        old_spect_flux = spectrum[..., sorted_input]
        if error is not None:
            error = error[..., sorted_input]
        old_spect_err = error

        # first we compute the input bin edges in wavenumber space
        if grid_width is None:
            old_spect_edges = compute_bin_edges(old_spect_wn)[0]
            old_spect_min = old_spect_edges[:-1]
            old_spect_max = old_spect_edges[1:]
        else:
            grid_width = grid_width[sorted_input]
            old_spect_min = old_spect_wn - grid_width / 2
            old_spect_max = old_spect_wn + grid_width / 2

        if self._in_wl:
            # convert input spectrum edges to wavelengths
            # must be flipped to remain sorted
            old_spect_min, old_spect_max = (
                1e4 / old_spect_max[::-1],
                1e4 / old_spect_min[::-1],
            )
            old_spect_flux = old_spect_flux[..., ::-1]  # must be flipped as well
            if old_spect_err is not None:
                old_spect_err = old_spect_err[..., ::-1]  # must be flipped as well

        # we compute the edges of the bins of the binner
        bin_mins = self._bin_centers - self._bin_widths / 2
        bin_maxes = self._bin_centers + self._bin_widths / 2

        # prepare outputs
        bin_spectrum = np.zeros(spectrum[..., 0].shape + self._bin_centers.shape)

        if error is not None:
            bin_error = np.zeros(spectrum[..., 0].shape + self._bin_centers.shape)
        else:
            bin_error = None

        for idx, (bin_min, bin_max) in enumerate(zip(bin_mins, bin_maxes, strict=True)):
            sum_spectrum = 0
            sum_noise = 0
            sum_weight = 0

            save_start = np.searchsorted(old_spect_max, bin_min, side="right")
            save_stop = np.searchsorted(old_spect_min[1:], bin_max, side="right")

            save_stop = min(save_stop, old_spect_min.shape[0] - 1)
            save_start = min(save_start, old_spect_min.shape[0] - 1)

            if (
                not bin_min <= old_spect_max[save_start]
                or not old_spect_min[save_stop] <= bin_max
            ):
                continue

            spect_min = old_spect_min[save_start : save_stop + 1]
            spect_max = old_spect_max[save_start : save_stop + 1]

            weight = (
                np.minimum(bin_max, spect_max) - np.maximum(spect_min, bin_min)
            ) / (bin_max - bin_min)

            sum_weight = np.sum(weight)

            sum_spectrum = np.sum(
                weight / sum_weight * old_spect_flux[..., save_start : save_stop + 1],
                axis=-1,
            )

            if error is not None:
                sum_noise = np.sum(
                    weight
                    * weight
                    * old_spect_err[..., save_start : save_stop + 1] ** 2,
                    axis=0,
                )

                sum_noise = np.sqrt(sum_noise / sum_weight / sum_weight)

            bin_spectrum[..., idx] = sum_spectrum

            if error is not None:
                bin_error[idx] = sum_noise

        return self._bin_centers, bin_spectrum, bin_error, self._bin_widths

    def generate_spectrum_output(
        self,
        model_output: ModelOutputType,
        output_size: t.Optional[OutputSize] = OutputSize.heavy,
    ) -> BinnedSpectrumType:
        """Generate spectrum output."""
        output = super().generate_spectrum_output(model_output, output_size=output_size)
        other_bin_centers = (
            1e4 / (self._bin_centers + self._bin_widths / 2)
            + 1e4 / (self._bin_centers - self._bin_widths / 2)
        ) / 2
        other_bin_widths = 1e4 / (self._bin_centers - self._bin_widths / 2) - 1e4 / (
            self._bin_centers + self._bin_widths / 2
        )
        if self._in_wl:
            output["binned_wngrid"] = other_bin_centers
            output["binned_wlgrid"] = self._bin_centers
            output["binned_wnwidth"] = other_bin_widths
            output["binned_wlwidth"] = self._bin_widths
        else:
            output["binned_wngrid"] = self._bin_centers
            output["binned_wlgrid"] = other_bin_centers
            output["binned_wnwidth"] = self._bin_widths
            output["binned_wlwidth"] = other_bin_widths

        return output

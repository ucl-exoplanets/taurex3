"""Convolution-aware flux binner for multi-instrument observations."""

import typing as t

import numpy as np
import numpy.typing as npt
from astropy.io import fits

from taurex import OutputSize
from taurex.util import compute_bin_edges
from taurex.util import create_grid_res
from taurex.util import wnwidth_to_wlwidth

from ..types import ModelOutputType
from .binner import BinDownType
from .binner import BinnedSpectrumType
from .binner import Binner
from .fluxbinner import FluxBinner


class FluxBinnerConv(Binner):
    """Bin to multiple wavelength grids with optional profile convolution."""

    def __init__(
        self,
        wlgrids: t.Sequence[npt.NDArray[np.float64]],
        wlgrid_widths: t.Sequence[npt.NDArray[np.float64]],
        broadening_profiles: t.Optional[t.Sequence[str]] = None,
        broadening_type: str = "stsci_fits",
        max_wlbroadening: t.Optional[float] = None,
        factor_cut: int = 5,
        wlres: float = 15000,
    ) -> None:
        super().__init__()

        if len(wlgrids) != len(wlgrid_widths):
            raise ValueError("wlgrids and wlgrid_widths must have the same length")

        self._wlgrids = [np.asarray(grid, dtype=np.float64) for grid in wlgrids]
        self._wlgrid_widths = [
            np.asarray(widths, dtype=np.float64) for widths in wlgrid_widths
        ]
        self._broadening_profiles = list(broadening_profiles or [])
        self._profile_type = broadening_type
        self._max_wlbroadening = max_wlbroadening
        self._factor_cut = factor_cut
        self._wlres = wlres

        self._wlgrid = np.concatenate(self._wlgrids)
        self._wlgrid_width = np.concatenate(self._wlgrid_widths)

        self.binners: t.List[FluxBinner] = []
        for grid, widths in zip(self._wlgrids, self._wlgrid_widths):
            sorter = np.argsort(grid)
            self.binners.append(FluxBinner(10000.0 / grid[sorter], widths[sorter]))

        self._profiles: t.List[npt.NDArray[np.float64]] = []
        self._grid_fbs: t.List[FluxBinner] = []
        if self._profile_type == "stsci_fits" and self._broadening_profiles:
            if len(self._broadening_profiles) != len(self._wlgrids):
                raise ValueError(
                    "broadening_profiles must match the number of wavelength grids"
                )
            self._profiles, self._grid_fbs = self.load_stsci_profiles(
                self._broadening_profiles
            )

    def load_stsci_profiles(
        self, files: t.Sequence[str]
    ) -> t.Tuple[t.List[npt.NDArray[np.float64]], t.List[FluxBinner]]:
        """Load STScI-style resolution profiles from FITS or text files."""

        profiles: t.List[npt.NDArray[np.float64]] = []
        grid_fbs: t.List[FluxBinner] = []

        for file_name, wlgrid in zip(files, self._wlgrids):
            try:
                with fits.open(file_name) as hdu:
                    science_data = hdu[1].data
                wavelength = np.asarray(science_data["WAVELENGTH"], dtype=np.float64)
                resolution = np.asarray(science_data["R"], dtype=np.float64)
            except OSError:
                science_data = np.loadtxt(file_name)
                wavelength = np.asarray(science_data[:, 0], dtype=np.float64)
                resolution = np.asarray(science_data[:, 1], dtype=np.float64)

            std = wavelength / resolution / 2.0
            native_grid = create_grid_res(
                self._wlres,
                wlgrid[0] - 10.0 * std[0],
                wlgrid[-1] + 10.0 * std[-1],
            )
            grid_fbs.append(FluxBinner(10000.0 / native_grid[:, 0], native_grid[:, 1]))

            sigma = np.interp(
                native_grid[:, 0],
                wavelength,
                std,
                left=std[0],
                right=std[-1],
            )
            if self._max_wlbroadening is not None:
                sigma = np.clip(sigma, a_min=1e-20, a_max=self._max_wlbroadening)
            else:
                sigma = np.clip(sigma, a_min=1e-20, a_max=None)
            profiles.append(sigma)

        return profiles, grid_fbs

    @staticmethod
    def gaussian(
        x: npt.NDArray[np.float64], mean: float, std: float
    ) -> npt.NDArray[np.float64]:
        """Compute a normalized Gaussian profile."""

        return (
            1.0
            / (np.sqrt(2.0 * np.pi) * std)
            * np.exp(-np.power((x - mean) / std, 2.0) / 2.0)
        )

    def low_res_convolved(
        self, binned_output: BinDownType, profile: npt.NDArray[np.float64]
    ) -> BinDownType:
        """Convolve a binned spectrum with a wavelength-dependent profile."""

        grid, flux, error, widths = binned_output
        convolved_flux = np.zeros_like(flux)

        for index, centre in enumerate(grid):
            std = profile[index]
            if index != len(grid) - 1:
                spacing = np.abs(grid[index + 1] - centre)
            else:
                spacing = np.abs(centre - grid[index - 1])

            window = max(1, int(self._factor_cut * std / spacing))
            start = max(0, index - window)
            stop = min(len(grid), index + window + 1)
            x = grid[start:stop]
            weights = self.gaussian(x, centre, std)
            convolved_flux[start:stop] += flux[index] * weights / np.sum(weights)

        return grid, convolved_flux, error, widths

    def _prepare_input(
        self,
        wngrid: npt.NDArray[np.float64],
        spectrum: npt.NDArray[np.float64],
        grid_width: t.Optional[npt.NDArray[np.float64]] = None,
        error: t.Optional[npt.NDArray[np.float64]] = None,
    ) -> BinDownType:
        if grid_width is not None:
            wlgrid_width = wnwidth_to_wlwidth(wngrid, grid_width)[::-1]
        else:
            wlgrid_width = None

        wlerror = error[::-1] if error is not None else None
        return 10000.0 / wngrid[::-1], spectrum[::-1], wlerror, wlgrid_width

    def bindown(
        self,
        wngrid: npt.NDArray[np.float64],
        spectrum: npt.NDArray[np.float64],
        grid_width: t.Optional[npt.NDArray[np.float64]] = None,
        error: t.Optional[npt.NDArray[np.float64]] = None,
    ) -> BinDownType:
        """Bind a native model spectrum to the configured instrument grids."""

        wlgrid, flux, wlerror, wlwidth = self._prepare_input(
            wngrid, spectrum, grid_width=grid_width, error=error
        )
        prepared: BinDownType = (wlgrid, flux, wlerror, wlwidth)

        wlgrids: t.List[npt.NDArray[np.float64]] = []
        spectra: t.List[npt.NDArray[np.float64]] = []
        errors: t.List[npt.NDArray[np.float64]] = []
        widths: t.List[npt.NDArray[np.float64]] = []

        for index, binner in enumerate(self.binners):
            working_output = prepared
            if self._profile_type == "stsci_fits" and self._profiles:
                working_output = self._grid_fbs[index].bindown(
                    10000.0 / prepared[0],
                    prepared[1],
                    grid_width=prepared[3],
                    error=prepared[2],
                )
                working_output = (
                    10000.0 / working_output[0][::-1],
                    working_output[1][::-1],
                    None if working_output[2] is None else working_output[2][::-1],
                    None if working_output[3] is None else working_output[3][::-1],
                )
                working_output = self.low_res_convolved(
                    working_output, self._profiles[index]
                )
                working_output = (
                    10000.0 / working_output[0][::-1],
                    working_output[1][::-1],
                    None if working_output[2] is None else working_output[2][::-1],
                    None if working_output[3] is None else working_output[3][::-1],
                )

            binned_output = binner.bindown(
                working_output[0],
                working_output[1],
                grid_width=working_output[3],
                error=working_output[2],
            )
            wlgrids.append(10000.0 / binned_output[0])
            spectra.append(binned_output[1])
            widths.append(wnwidth_to_wlwidth(binned_output[0], binned_output[3]))
            if binned_output[2] is not None:
                errors.append(binned_output[2])

        merged_wlgrid = np.concatenate(wlgrids)
        merged_spectrum = np.concatenate(spectra)
        merged_error = np.concatenate(errors) if errors else None
        merged_widths = np.concatenate(widths)

        return merged_wlgrid, merged_spectrum, merged_error, merged_widths

    def generate_spectrum_output(
        self,
        model_output: ModelOutputType,
        output_size: t.Optional[OutputSize] = OutputSize.heavy,
    ) -> BinnedSpectrumType:
        """Generate a TauREx-style spectrum output dictionary."""

        output = super().generate_spectrum_output(model_output, output_size=output_size)
        output["binned_wngrid"] = 10000.0 / self._wlgrid
        output["binned_wlgrid"] = self._wlgrid
        output["binned_wnwidth"] = compute_bin_edges(output["binned_wngrid"])[-1]
        output["binned_wlwidth"] = self._wlgrid_width
        return output

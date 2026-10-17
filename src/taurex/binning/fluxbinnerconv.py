"""Convolution-aware flux binner for multi-instrument observations."""

import typing as t

import numpy as np
import numpy.typing as npt
from astropy.io import fits
from numpy.polynomial import chebyshev

from taurex import OutputSize
from taurex.util import compute_bin_edges
from taurex.util import create_grid_res
from taurex.util import wnwidth_to_wlwidth

from ..types import ModelOutputType
from .binner import BinDownType
from .binner import BinnedSpectrumType
from .binner import Binner
from .fluxbinner import FluxBinner


def lsf_mode(entry: t.Optional[str]) -> str:
    """Classify one the braodning entries as: "none", "function" or "file"."""
    if entry is None or str(entry).strip().lower() in ("", "none"):
        return "none"
    if str(entry).strip().lower() == "function":
        return "functional"
    return "file"


class FluxBinnerConv(Binner):
    """Bin to multiple wavelength grids with optional profile convolution."""

    def __init__(
        self,
        wlgrids: t.Sequence[npt.NDArray[np.float64]],
        wlgrid_widths: t.Sequence[npt.NDArray[np.float64]],
        broadening_profiles: t.Optional[t.Sequence[str]] = None,
        broadening_type: str = "stsci_fits",
        wlshift: t.Optional[t.Union[float, t.Sequence[float]]] = 0.0,
        max_wlbroadening: t.Optional[float] = None,
        broadening_coeffs=None,
        broadening_basis: str = "chebyshev",
        factor_cut: int = 5,
        wlres: float = 15000,
        fwhm_to_sigma: float = 2.355,
        max_wlshift: float = 0.0,
    ) -> None:
        """Initialise with wavelength grids and optional broadening profiles."""
        super().__init__()

        if len(wlgrids) != len(wlgrid_widths):
            raise ValueError("wlgrids and wlgrid_widths must have the same length")

        self._wlshifts = self._normalize_wlshift(wlshift, len(wlgrids))
        self._max_wlshift = max_wlshift
        self._unshifted_wlgrids = [np.asarray(grid, dtype=np.float64) for grid in wlgrids]
        self._binner_shifts = list(self._wlshifts)
        self._wlgrids = [
            np.asarray(grid, dtype=np.float64) + shift
            for grid, shift in zip(wlgrids, self._wlshifts, strict=False)
        ]
        self._wlgrid_widths = [
            np.asarray(widths, dtype=np.float64) for widths in wlgrid_widths
        ]
        self._broadening_profiles = list(broadening_profiles or [])
        self._profile_type = broadening_type
        self._max_wlbroadening = max_wlbroadening
        self._factor_cut = factor_cut
        self._wlres = wlres
        self._broadening_coeffs = broadening_coeffs
        self._broadening_basis = broadening_basis
        self._fwhm_to_sigma = fwhm_to_sigma

        self._wlgrid = np.concatenate(self._wlgrids)
        self._wlgrid_width = np.concatenate(self._wlgrid_widths)

        self.binners: t.List[FluxBinner] = []
        for grid, widths in zip(self._wlgrids, self._wlgrid_widths, strict=False):
            sorter = np.argsort(grid)
            self.binners.append(
                FluxBinner(wlgrid=grid[sorter], wlgrid_width=widths[sorter])
            )

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
        elif self._profile_type == "polynomial":
            if self._max_wlbroadening is None:
                raise ValueError(
                    "max_wlbroadening is required when broadening_type='polynomial'"
                )
            for wlgrid, widths in zip(self._wlgrids, self._wlgrid_widths, strict=True):
                pad = self._factor_cut * self._max_wlbroadening + self._max_wlshift
                native = create_grid_res(
                    self._wlres,
                    wlgrid[0] - widths[0] / 2 - pad,
                    wlgrid[-1] + widths[-1] / 2 + pad,
                )
                self._grid_fbs.append(
                    FluxBinner(
                        wngrid=10000.0 / native[:, 0],
                        wngrid_width=10000.0 * native[:, 1] / native[:, 0] ** 2,
                    )
                )
                self._profiles.append(None)

    @staticmethod
    def _normalize_wlshift(
        wlshift: t.Optional[t.Union[float, t.Sequence[float]]],
        grid_count: int,
    ) -> t.List[float]:
        if wlshift is None:
            return [0.0] * grid_count

        if isinstance(wlshift, list) and len(wlshift) == grid_count:
            return wlshift 

        if isinstance(wlshift, (list, tuple, np.ndarray)):
            shifts = [float(shift) for shift in wlshift]
            if len(shifts) != grid_count:
                raise ValueError("wlshift must match the number of wavelength grids")
            return shifts

        return [float(wlshift)] * grid_count

    def sigma(self, index):
        """Gaussian broadening sigma on the convolution grid of instrument `index`."""
        coeffs = None if self._broadening_coeffs is None else self._broadening_coeffs[index]
        profile = self._profiles[index]
        if coeffs is None:
            if profile is None:
                raise ValueError(f"spectrum {index + 1} has a function LSF but no broadening_order")
            return profile

        wl = 10000.0 / self._grid_fbs[index]._bin_centers[::-1]

        if self._broadening_basis == "sigma_poly":
            # legacy in sigma directly
            poly = np.polynomial.polynomial.polyval(wl, coeffs)
            if profile is not None:
                sigma = profile + poly
            else:
                sigma = poly
            return np.clip(sigma, 1e-20, self._max_wlbroadening)

        if self._broadening_basis == "resolution_poly":
            # legacy in Resolution
            resolution = np.polynomial.polynomial.polyval(wl, coeffs)
            if profile is not None:
                resolution = resolution + wl / (self._fwhm_to_sigma * profile)
            with np.errstate(divide="ignore", invalid="ignore"):
                sigma = wl / (
                    self._fwhm_to_sigma * np.clip(resolution, a_min=1, a_max=self._wlres)
                )
            sigma = np.nan_to_num(sigma, nan=1e-20, posinf=np.inf, neginf=1e-20)
            return np.clip(sigma, 1e-20, self._max_wlbroadening)

        # default is log-space Chebyshev
        x = 2.0 * (wl - wl[0]) / (wl[-1] - wl[0]) - 1.0
        poly = np.exp(chebyshev.chebval(x, coeffs))
        if self._broadening_basis == "chebyshev_linear":
            if profile is not None:
                sigma = profile * np.maximum(1.0 + poly, 0.05)
            else:
                sigma = wl / (self._fwhm_to_sigma * np.maximum(poly, 1.0))
        else:
            if profile is not None:
                sigma = profile * poly  # this handles departure from calibration
            else:
                # this handles direct fitting of resolution
                sigma = wl / (self._fwhm_to_sigma * poly)
        return np.clip(sigma, 0.5 * np.gradient(wl), self._max_wlbroadening)

    def load_stsci_profiles(
        self, files: t.Sequence[str]
    ) -> t.Tuple[t.List[npt.NDArray[np.float64]], t.List[FluxBinner]]:
        """Load STScI-style resolution profiles from FITS or text files."""
        profiles: t.List[npt.NDArray[np.float64]] = []
        grid_fbs: t.List[FluxBinner] = []

        for file_name, wlgrid, widths in zip(files, self._wlgrids, self._wlgrid_widths, strict=True):
            mode = lsf_mode(file_name)
            if mode == "none":  # binned only, no convolution
                profiles.append(None)
                grid_fbs.append(None)
                continue
            if mode == "functional":  # unknown LSF, so use broadening_order
                if self._max_wlbroadening is None:
                    raise ValueError("max_wlbroadening is required for a functional LSF")
                pad = self._factor_cut * self._max_wlbroadening + self._max_wlshift
                native_grid = create_grid_res(self._wlres, wlgrid[0] - widths[0] / 2 - pad, wlgrid[-1] + widths[-1] / 2 + pad,)
                ww = wnwidth_to_wlwidth(native_grid[:, 0], native_grid[:, 1])
                grid_fbs.append(FluxBinner(10000.0 / native_grid[:, 0], ww))
                profiles.append(None)
                continue
            try:
                with fits.open(file_name) as hdu:
                    science_data = hdu[1].data
                wavelength = np.asarray(science_data["WAVELENGTH"], dtype=np.float64)
                resolution = np.asarray(science_data["R"], dtype=np.float64)
            except OSError:
                science_data = np.loadtxt(file_name)
                wavelength = np.asarray(science_data[:, 0], dtype=np.float64)
                resolution = np.asarray(science_data[:, 1], dtype=np.float64)

            std = wavelength / (resolution * self._fwhm_to_sigma) # Convert FWHM to sigma 2.355 is default as JWST
            edge_std = np.interp([wlgrid[0], wlgrid[-1]], wavelength, std)
            pad = self._factor_cut * 1.5 * edge_std + self._max_wlshift ## Larger padding
            native_grid = create_grid_res(
                self._wlres,
                wlgrid[0] - widths[0] / 2 - pad[0],
                wlgrid[-1] + widths[-1] / 2 + pad[1],)
            # grid_fbs.append(FluxBinner(10000.0 / native_grid[:, 0],
            #     native_grid[:, 1]))  # It was this before, must be a forgotten error!
            ww = wnwidth_to_wlwidth(native_grid[:, 0], native_grid[:, 1])
            grid_fbs.append(FluxBinner(10000.0 / native_grid[:, 0], ww))

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

    def _shifted_binner(self, index: int) -> FluxBinner:
        """Rebuild binner if its shift changed."""
        shift = self._wlshifts[index]
        if shift != self._binner_shifts[index]:
            grid = self._unshifted_wlgrids[index] + shift
            self.binners[index] = FluxBinner(
                wlgrid=grid, wlgrid_width=self._wlgrid_widths[index]
            )
            self._wlgrids[index] = grid
            self._wlgrid = np.concatenate(self._wlgrids)  # outputs follow the shift
            self._binner_shifts[index] = shift
        return self.binners[index]

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
        convolved_flux = np.zeros(flux.shape, dtype=np.float64)

        dwl = np.gradient(grid)
        lo = np.searchsorted(grid, grid - self._factor_cut * profile, side="left")
        hi = np.searchsorted(grid, grid + self._factor_cut * profile, side="right")

        for index, centre in enumerate(grid):
            start = int(lo[index])
            stop = max(int(hi[index]), start + 1)
            weights = (
                self.gaussian(grid[start:stop], centre, profile[index])
                * dwl[start:stop]
            )
            weights /= weights.sum()
            convolved_flux[..., index] = np.sum(
                flux[..., start:stop] * weights, axis=-1
            )

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
        """Bin a native model spectrum onto the configured instrument grids."""
        wlgrids = []
        spectra = []
        errors = []
        widths = []

        for index in range(len(self.binners)):
            binner = self._shifted_binner(index)
            wn, flux, err, wnwidth = wngrid, spectrum, error, grid_width

            if self._grid_fbs and self._grid_fbs[index] is not None:
                wn, flux, err, wnwidth = self._grid_fbs[index].bindown(
                    wn, flux, grid_width=wnwidth, error=err
                )
                conv_wl = 10000.0 / wn[::-1]
                flux = self.low_res_convolved(
                    (conv_wl, flux[..., ::-1], None, None), self.sigma(index)
                )[1][..., ::-1]

            binned_output = binner.bindown(wn, flux, grid_width=wnwidth, error=err)
            wlgrids.append(binned_output[0])
            spectra.append(binned_output[1])
            widths.append(binned_output[3])
            if binned_output[2] is not None:
                errors.append(binned_output[2])

        merged_wlgrid = np.concatenate(wlgrids)
        merged_spectrum = np.concatenate(spectra, axis=-1)
        merged_error = np.concatenate(errors, axis=-1) if error is not None else None
        merged_widths = np.concatenate(widths)

        return merged_wlgrid, merged_spectrum, merged_error, merged_widths

    def bindown_old(
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
                wn_grid = 10000.0 / prepared[0]
                conv_grid_width = (
                    wn_grid**2 * prepared[3] / 10000.0
                    if prepared[3] is not None
                    else None
                )
                working_output = self._grid_fbs[index].bindown(
                    wn_grid,
                    prepared[1],
                    grid_width=conv_grid_width,
                    error=prepared[2],
                )
                working_output = (
                    10000.0 / working_output[0][::-1],
                    working_output[1][::-1],
                    None if working_output[2] is None else working_output[2][::-1],
                    None if working_output[3] is None else working_output[3][::-1],
                )
                working_output = self.low_res_convolved(
                    working_output, self.sigma(index)
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
            wlgrids.append(binned_output[0])
            spectra.append(binned_output[1])
            widths.append(binned_output[3])
            if binned_output[2] is not None:
                errors.append(binned_output[2])

        merged_wlgrid = np.concatenate(wlgrids)
        merged_spectrum = np.concatenate(spectra, axis=-1)
        merged_error = (
            None
            if not errors or any(e is None for e in errors)
            else np.concatenate(errors, axis=-1)
        )
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

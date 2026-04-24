"""Observed spectra with per-instrument systematic corrections."""

import copy
import typing as t

import numpy as np
import numpy.typing as npt

from taurex.binning import FluxBinnerConv
from taurex.util import compute_bin_edges
from taurex.util import wnwidth_to_wlwidth

from .spectrum import BaseSpectrum


class _OffsetSpectrumBase(BaseSpectrum):
    """Shared implementation for multi-spectrum observations."""

    def __init__(
        self,
        path_spectra: t.Optional[t.Sequence[str]] = None,
        offsets: t.Optional[t.Sequence[float]] = None,
        slopes: t.Optional[t.Sequence[float]] = None,
        error_scale: t.Optional[t.Sequence[float]] = None,
        slope_type: t.Optional[str] = "linear",
    ) -> None:
        super().__init__(self.__class__.__name__)

        self.path_spectra = list(path_spectra or [])
        self.slope_type = slope_type
        self.offsets = self._normalize_parameter(offsets, 0.0, "offsets")
        self.slopes = self._normalize_parameter(slopes, 0.0, "slopes")
        self.escale = self._normalize_parameter(error_scale, 1.0, "error_scale")

        self._raw = [self._load_spectrum(path) for path in self.path_spectra]
        self._raw = [self._sort_spectrum(spectrum) for spectrum in self._raw]

        self._raw_bin_edges: t.List[npt.NDArray[np.float64]] = []
        self._raw_bin_widths: t.List[npt.NDArray[np.float64]] = []
        self._raw_wnwidths: t.List[npt.NDArray[np.float64]] = []
        self._bin_edges = np.array([], dtype=np.float64)
        self._wnwidths = np.array([], dtype=np.float64)

        self._process_spectra()
        self._obs_spectrum = self._combine_raw_spectra()
        self.generate_offset_fitting_params()

    def _normalize_parameter(
        self,
        values: t.Optional[t.Sequence[float]],
        default: float,
        parameter_name: str,
    ) -> t.List[float]:
        count = len(self.path_spectra)
        if values is None:
            return [default] * count

        normalized = list(values)
        if len(normalized) == 0:
            return [default] * count
        if len(normalized) != count:
            raise ValueError(
                f"{parameter_name} must have the same length as path_spectra"
            )
        return normalized

    @staticmethod
    def _load_spectrum(path: str) -> npt.NDArray[np.float64]:
        data = np.loadtxt(path)
        if data.ndim == 1:
            data = np.array([data], dtype=np.float64)
        if data.ndim != 2 or data.shape[1] not in (3, 4):
            raise ValueError("spectra must be a 2D array with 3 or 4 columns")
        return np.asarray(data, dtype=np.float64)

    @staticmethod
    def _sort_spectrum(spectrum: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        sorter = np.argsort(spectrum[:, 0])
        return spectrum[sorter, :]

    def _process_spectra(self) -> None:
        if not self._raw:
            return

        for spectrum in self._raw:
            if spectrum.shape[1] == 3:
                bin_edges, bin_widths = compute_bin_edges(spectrum[:, 0])
            else:
                bin_widths = spectrum[:, 3]
                obs_wl = spectrum[:, 0]
                bin_edges = np.empty(bin_widths.size * 2, dtype=np.float64)
                bin_edges[0::2] = obs_wl - bin_widths / 2.0
                bin_edges[1::2] = obs_wl + bin_widths / 2.0

            self._raw_bin_edges.append(bin_edges)
            self._raw_bin_widths.append(bin_widths)
            self._raw_wnwidths.append(wnwidth_to_wlwidth(spectrum[:, 0], bin_widths))

        self._bin_edges = np.concatenate(self._raw_bin_edges)
        self._wnwidths = np.concatenate(self._raw_wnwidths)

    def _combine_raw_spectra(self) -> npt.NDArray[np.float64]:
        if not self._raw:
            return np.empty((0, 4), dtype=np.float64)

        normalized = [
            np.column_stack([spectrum[:, :3], widths])
            for spectrum, widths in zip(self._raw, self._raw_bin_widths)
        ]
        return np.vstack(normalized)

    @property
    def rawData(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Data read from file."""
        return self._obs_spectrum

    @property
    def wavelengthGrid(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Wavelength grid in microns."""
        return self.rawData[:, 0]

    @property
    def binEdges(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Bin edges in wavenumber space."""
        return 10000.0 / self._bin_edges

    @property
    def binWidths(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Bin widths in wavenumber space."""
        return self._wnwidths

    @property
    def spectrum(self) -> npt.NDArray[np.float64]:
        """Spectrum with per-instrument systematic corrections applied."""

        fluxes = []
        for spectrum, offset, slope in zip(self._raw, self.offsets, self.slopes):
            corrected = copy.deepcopy(spectrum[:, 1])
            wavelengths = spectrum[:, 0]

            if self.slope_type in ("linear", None):
                corrected = (
                    corrected + offset + slope * (wavelengths - np.mean(wavelengths))
                )
            elif self.slope_type == "log":
                corrected = (
                    corrected
                    + offset
                    + slope * np.power(10.0, wavelengths - np.mean(wavelengths))
                )
            else:
                raise ValueError(f"Unsupported slope_type {self.slope_type}")

            fluxes.append(corrected)

        return np.concatenate(fluxes) if fluxes else np.array([], dtype=np.float64)

    @property
    def errorBar(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Error bars with per-instrument scaling applied."""

        errors = [
            scale * spectrum[:, 2] for spectrum, scale in zip(self._raw, self.escale)
        ]
        return np.concatenate(errors) if errors else np.array([], dtype=np.float64)

    def generate_offset_fitting_params(self) -> None:
        """Create fittable parameters for each input spectrum."""

        bounds = (-0.001, 0.001)
        for index in range(len(self.offsets)):
            point_num = index + 1

            def read_offset(self, index=index):
                return self.offsets[index]

            def write_offset(self, value, index=index):
                self.offsets[index] = value

            self.add_fittable_param(
                f"Offset_{point_num}",
                f"$Offset_{point_num}$",
                read_offset,
                write_offset,
                "linear",
                False,
                bounds,
            )

            def read_escale(self, index=index):
                return self.escale[index]

            def write_escale(self, value, index=index):
                self.escale[index] = value

            self.add_fittable_param(
                f"EScale_{point_num}",
                f"$Escale_{point_num}$",
                read_escale,
                write_escale,
                "linear",
                False,
                bounds,
            )

            def read_slope(self, index=index):
                return self.slopes[index]

            def write_slope(self, value, index=index):
                self.slopes[index] = value

            self.add_fittable_param(
                f"Slope_{point_num}",
                f"$Slope_{point_num}$",
                read_slope,
                write_slope,
                "linear",
                False,
                bounds,
            )


class OffsetSpectra(_OffsetSpectrumBase):
    """Multiple observed spectra with offset, slope, and error-scale parameters."""

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Input keywords for the basic multi-spectrum implementation."""

        return ("spectra_w_offsets", "observation_w_offsets")


class OffsetSpectraCont(_OffsetSpectrumBase):
    """Multiple observed spectra with optional instrument broadening."""

    def __init__(
        self,
        path_spectra: t.Optional[t.Sequence[str]] = None,
        offsets: t.Optional[t.Sequence[float]] = None,
        slopes: t.Optional[t.Sequence[float]] = None,
        error_scale: t.Optional[t.Sequence[float]] = None,
        slope_type: t.Optional[str] = "linear",
        broadening_type: str = "stsci_fits",
        broadening_profiles: t.Optional[t.Sequence[str]] = None,
        wlshift: float = 0.0,
        max_wlbroadening: float = 0.1,
        factor_cut: int = 5,
        wlres: float = 15000,
    ) -> None:
        self._wlshift = wlshift
        self._broadening_profiles = list(broadening_profiles or [])
        self._profile_type = broadening_type
        self._max_wlbroadening = max_wlbroadening
        self._factor_cut = factor_cut
        self._wlres = wlres

        super().__init__(
            path_spectra=path_spectra,
            offsets=offsets,
            slopes=slopes,
            error_scale=error_scale,
            slope_type=slope_type,
        )

    def create_binner(self) -> FluxBinnerConv:
        """Create a multi-grid binner for the instrument spectra."""

        return FluxBinnerConv(
            wlgrids=[spectrum[:, 0] for spectrum in self._raw],
            wlgrid_widths=self._raw_bin_widths,
            broadening_profiles=self._broadening_profiles,
            broadening_type=self._profile_type,
            max_wlbroadening=self._max_wlbroadening,
            factor_cut=self._factor_cut,
            wlres=self._wlres,
        )

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Input keywords for the instrument-systematics implementation."""

        return ("spectra_instr", "observation_instr")

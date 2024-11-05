"""Module dealing with observed lightcurves."""

import typing as t

import numpy as np
import numpy.typing as npt

from taurex.binning.lightcurvebinner import LightcurveBinner
from taurex.model.lightcurve.lightcurvedata import LCDataType, LightCurveData
from taurex.output import OutputGroup
from taurex.types import PathLike

from .spectrum import BaseSpectrum


class ObservedLightCurve(BaseSpectrum):
    """Loads an observed lightcurve from a pickle file."""

    def __init__(self, filename: t.Optional[PathLike] = None):
        """Initialize lightcurve.

        Parameters
        ----------
        filename:
            Filename of lightcurve pickle data, by default None

        """
        super().__init__("observed_lightcurve")

        import pickle  # noqa: S403

        with open(filename, "rb") as f:
            lc_data = pickle.load(f, encoding="latin1")  # noqa: S301
        # new version
        self.obs_spectrum = np.empty(shape=(len(lc_data["obs_spectrum"][:, 0]), 4))

        # new version
        self.obs_spectrum[:, 0] = lc_data["obs_spectrum"][:, 0]
        self.obs_spectrum[:, 1] = lc_data["obs_spectrum"][:, 1]
        self.obs_spectrum[:, 2] = lc_data["obs_spectrum"][:, 2]
        self.obs_spectrum[:, 3] = lc_data["obs_spectrum"][:, 3]

        self._spec, self._std = self._load_data_file(lc_data)

    def create_binner(self) -> LightcurveBinner:
        """Creates the appropriate binning object."""
        return LightcurveBinner()

    def _load_data_file(
        self, lc_data: LCDataType
    ) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Load and combine data from different instruments.

        Parameters
        ----------
        lc_data:
            Lightcurve data

        Returns
        -------
        combine_lc:
            Combined lightcurve.
        combine_lc_std:
            Combined lightcurve uncertainty.


        """
        raw_data = []
        data_std = []
        wngrid_min = []

        for ins in LightCurveData.availableInstruments:
            # new version
            # raw data includes data and datastd.
            if ins in lc_data:
                wngrid_min.append(lc_data[ins]["wl_grid"].min())
                raw_data.append(lc_data[ins]["data"][:, :, 0])
                data_std.append(lc_data[ins]["data"][:, :, 1])
        wngrid_min, raw_data, data_std = (
            list(t)
            for t in zip(
                *sorted(
                    zip(wngrid_min, raw_data, data_std),
                    key=lambda x: x[0],
                    reverse=True,
                )
            )
        )
        return np.concatenate(raw_data), np.concatenate(data_std)

    @property
    def spectrum(self) -> npt.NDArray[np.float64]:
        """Return a Light curve `spectrum`.

        Spectrum is not a true spectrum but in the context of Taurex it is
        seen as one to a retrieval.

        The lightcurve spectrum comes in the form of multiple lightcurves
        stuck together into
        one long spectrum. The number of lightcurves is equal to the number of
        bins in :func:`wavelengthGrid`.

        """
        return self._spec

    @property
    def rawData(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Raw lightcurve data read from file

        Returns
        -------
        lc_data : :obj:`array`

        """
        self.obs_spectrum

    @property
    def wavelengthGrid(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Returns wavelength grid in microns

        Returns
        -------
        wlgrid : :obj:`array`

        """
        return self.obs_spectrum[:, 0]

    @property
    def binEdges(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Returns bin edges for wavelength grid.

        Returns
        -------
        out : :obj:`array`

        """
        return self.obs_spectrum[:, 3]

    @property
    def binWidths(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Widths for each bin in wavelength grid.

        Returns
        -------
        out : :obj:`array`

        """
        return np.zeros(2)

    @property
    def errorBar(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Uncertainty of lightcurve spectrum.

        Returns
        -------
        err : :obj:`array`
            Error at each point in lightcurve spectrum

        """
        return self._std

    def write(self, output: OutputGroup) -> OutputGroup:
        """Write to output group."""
        output.write_array("wlgrid", self.wavelengthGrid)
        output.write_array("spectrum", self.obs_spectrum[:, 1])
        output.write_array("lightcurve", self.spectrum)
        output.write_array("binedges", self.binEdges)
        output.write_array("binwidths", self.binWidths)
        output.write_array("errorbars", self.obs_spectrum[:, 2])
        output.write_array("lightcurve_errorbars", self.errorBar)

        return output

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Input keywords for this class."""
        return ("lightcurve",)

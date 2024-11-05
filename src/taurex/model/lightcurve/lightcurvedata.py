"""Lightcurve data class."""
import logging
import typing as t

import numpy as np
import numpy.typing as npt

from taurex.log import Logger
from taurex.output import OutputGroup, Writeable


class LightCurveInstrumentData(t.TypedDict):
    """Lightcurve data for a single instrument."""

    data: npt.NDArray[np.float64]
    time_series: npt.NDArray[np.float64]
    ld_coeff: npt.NDArray[np.float64]


InstrumentType = t.Literal["wfc3", "spitzer", "stis", "twinkle"]

LCDataType = t.Dict[InstrumentType, LightCurveInstrumentData]


class LightCurveData(Logger, Writeable):
    """Class holding data for a lightcurve from an instrument."""

    availableInstruments = ["wfc3", "spitzer", "stis", "twinkle"]  # noqa: N815

    @classmethod
    def fromInstrumentName(  # noqa: N802
        cls,
        name: InstrumentType,
        lc_data: LCDataType,
    ):
        log = logging.getLogger()
        if name.lower() in ("wfc3",):
            return cls(lc_data, "wfc3", (1.1, 1.8))
        elif name.lower() in ("spitzer",):
            return cls(lc_data, "spitzer", (3.4, 8.2))
        elif name.lower() in ("stis",):
            return cls(lc_data, "stis", (0.3, 1.0))
        elif name.lower() in ("twinkle",):
            return cls(lc_data, "twinkle", (0.4, 4.5))
        else:
            log.error(
                "LightCurve of instrument %s not recognized" " or implemented", name
            )
            raise KeyError

    def __init__(
        self,
        lc_data: LCDataType,
        instrument_name: InstrumentType,
        wavelength_region: t.Tuple[float, float],
    ):
        """Initialize.

        Parameters
        ----------
        lc_data:
            Lightcurve data
        instrument_name:
            Name of instrument
        wavelength_region:
            Wavelength region of instrument

        Raises
        ------
        KeyError
            If instrument name not found in lightcurve data


        """
        super().__init__(self.__class__.__name__)
        self._instrument_name = instrument_name
        # new version
        if self._instrument_name not in lc_data:
            self.error(
                "Instrument with key %s not found in pickled lightcurve" " file",
                self._instrument_name,
            )
            raise KeyError()

        self._wavelength_region = wavelength_region
        self._load_data(lc_data)

    def _load_data(self, lc_data: LCDataType):
        """Load data from lightcurve data."""
        # new
        self._time_series = lc_data[self._instrument_name]["time_series"]

        # new
        self._raw_data = lc_data[self._instrument_name]["data"][:, :, 0]
        self._data_std = lc_data[self._instrument_name]["data"][:, :, 1]

        self._max_nfactor = np.max(self._raw_data, axis=1)
        self._min_nfactor = np.min(self._raw_data, axis=1)

    @property
    def instrumentName(self) -> InstrumentType:  # noqa: N802
        """Instrument name."""
        return self._instrument_name

    @property
    def wavelengthRegion(self) -> t.Tuple[float, float]:  # noqa: N802
        """Wavelength region of instrument."""
        return self._wavelength_region

    @property
    def timeSeries(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Time series of data."""
        return self._time_series

    @property
    def rawData(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Raw data."""
        return self._raw_data

    @property
    def dataError(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Lightcurve uncertainty."""
        return self._data_std

    @property
    def minNFactors(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Minimum number of factors."""
        return self._min_nfactor

    @property
    def maxNFactors(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Maximum number of factors."""
        return self._max_nfactor

    def write(self, output: OutputGroup) -> OutputGroup:
        """Write to output group."""
        lc_grp = output.create_group(self.instrumentName)

        lc_grp.write_array("raw_data", self.rawData)
        lc_grp.write_array("data_error", self.dataError)
        lc_grp.write_array("min_n_factors", self.minNFactors)
        lc_grp.write_array("max_n_factors", self.maxNFactors)
        lc_grp.write_array("time_series", self.timeSeries)
        lc_grp.write_scalar("min_wavelength", self.wavelengthRegion[0])
        lc_grp.write_scalar("max_wavelength", self.wavelengthRegion[1])

        return lc_grp

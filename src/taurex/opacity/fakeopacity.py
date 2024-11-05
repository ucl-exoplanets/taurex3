"""Opacity for testing purposes."""
import typing as t

import numpy as np
import numpy.typing as npt

from taurex.util import create_grid_res

from . import InterpolatingOpacity


class FakeOpacity(InterpolatingOpacity):
    """Fake opacity for testing purposes."""

    def __init__(
        self,
        molecule_name: str,
        wn_res: t.Optional[float] = 15000,
        wn_size: t.Optional[t.Tuple[float, float]] = (300, 30000),
        num_p: t.Optional[int] = 20,
        num_t: t.Optional[int] = 27,
    ):
        super().__init__("FAKE")
        self._molecule_name = molecule_name
        self._wavenumber_grid = create_grid_res(wn_res, *wn_size)[:, 0]
        self._xsec_grid = np.random.rand(num_p, num_t, self._wavenumber_grid.shape[0])

        self._temperature_grid = np.linspace(100, 10000, num_t)
        self._pressure_grid = np.logspace(-6, 6, num_p)

    @property
    def moleculeName(self) -> str:  # noqa: N802
        """Name of molecule."""
        return self._molecule_name

    @property
    def xsecGrid(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Opacity grid."""
        return self._xsec_grid

    @property
    def wavenumberGrid(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Wavenumber grid."""
        return self._wavenumber_grid

    @property
    def temperatureGrid(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Temperature grid."""
        return self._temperature_grid

    @property
    def pressureGrid(self) -> npt.NDArray[np.float64]:  # noqa: N802
        """Pressure grid."""
        return self._pressure_grid

"""Base class for ktables"""
import typing as t

import numpy as np
import numpy.typing as npt

if t.TYPE_CHECKING:
    from ..opacity import Opacity

    _Base = Opacity
else:
    _Base = object


class KTable(_Base):
    """A mixin class to represent a ktable."""

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Quadrature weights."""
        raise NotImplementedError

    def opacity(
        self,
        temperature: float,
        pressure: float,
        wngrid: t.Optional[npt.NDArray[np.float64]] = None,
    ) -> npt.NDArray[np.float64]:
        """Opacity at given temperature and pressure.

        Parameters
        ----------
        temperature:
            Temperature in K
        pressure:
            Pressure in Pa
        wngrid:
            Wavenumber grid to interpolate to

        Returns
        -------
        np.ndarray:
            Interpolated opacity

        """
        from scipy.interpolate import interp1d

        wngrid_filter = slice(None)
        if wngrid is not None:
            wngrid_filter = np.where(
                (self.wavenumberGrid >= wngrid.min())
                & (self.wavenumberGrid <= wngrid.max())
            )[0]
        orig = self.compute_opacity(temperature, pressure, wngrid_filter).reshape(
            -1, len(self.weights)
        )

        if wngrid is None or np.array_equal(
            self.wavenumberGrid.take(wngrid_filter), wngrid
        ):
            return orig
        else:
            f = interp1d(
                self.wavenumberGrid[wngrid_filter],
                orig,
                axis=0,
                copy=False,
                bounds_error=False,
                fill_value=(orig[0], orig[-1]),
                assume_sorted=True,
            )
            return f(wngrid).reshape(-1, len(self.weights))

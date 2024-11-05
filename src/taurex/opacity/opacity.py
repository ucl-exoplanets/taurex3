"""Base class for computing opacities."""
import typing as t

import numpy as np
import numpy.typing as npt

from taurex.cache import GlobalCache
from taurex.core import Citable
from taurex.log import Logger


class Opacity(Logger, Citable):
    """This is the base class for computing opactities."""

    @classmethod
    def discover(cls) -> t.List[t.Tuple[str, t.Any]]:
        """Class method, used to discover molecular opacities.

        Should return opacities that are available in the current
        environment.

        """
        raise NotImplementedError

    @classmethod
    def priority(cls) -> int:
        """Priority of the opacity during discovery."""
        return 100

    def __init__(self, name: str) -> None:
        """Initialize opacity.

        Parameters
        ----------
        name: str
            Name of opacity for logging.

        """
        super().__init__(name)

    @property
    def resolution(self) -> float:
        """Resolution of opacity."""
        raise NotImplementedError

    @property
    def moleculeName(self) -> str:  # noqa: N802
        """Name of molecule."""
        raise NotImplementedError

    @property
    def wavenumberGrid(self) -> npt.NDArray[np.float64]:  # noqa: N802
        raise NotImplementedError

    @property
    def temperatureGrid(self) -> npt.NDArray[np.float64]:  # noqa: N802
        raise NotImplementedError

    @property
    def pressureGrid(self) -> npt.NDArray[np.float64]:  # noqa: N802
        raise NotImplementedError

    def compute_opacity(
        self,
        temperature: float,
        pressure: float,
        wngrid: t.Optional[npt.NDArray[np.float64]] = None,
    ) -> npt.NDArray[np.float64]:
        """Must return in units of cm2."""
        raise NotImplementedError

    def opacity(
        self,
        temperature: float,
        pressure: float,
        wngrid: t.Optional[npt.NDArray[np.float64]] = None,
    ) -> npt.NDArray[np.float64]:
        """Compute the opacity for a given temperature and pressure.

        Parameters
        ----------
        temperature: float
            Temperature in K
        pressure: float
            Pressure in Pa
        wngrid: :obj:`array`, optional
            Wavenumber grid to restrict to.

        Returns
        -------
        :obj:`array`
            Opacity array


        """
        if wngrid is None:
            wngrid_filter = slice(None)
        else:
            wngrid_filter = np.where(
                (self.wavenumberGrid >= wngrid.min())
                & (self.wavenumberGrid <= wngrid.max())
            )[0]

        orig = self.compute_opacity(temperature, pressure, wngrid_filter)

        if wngrid is None or np.array_equal(
            self.wavenumberGrid.take(wngrid_filter), wngrid
        ):
            return orig
        else:
            hold_method = GlobalCache()["opacity_hold"]
            if hold_method:
                return np.interp(wngrid, self.wavenumberGrid[wngrid_filter], orig)
            else:
                return np.interp(
                    wngrid,
                    self.wavenumberGrid[wngrid_filter],
                    orig,
                    left=0.0,
                    right=0.0,
                )

    def opacityCitation(self) -> t.List[str]:  # noqa: N802
        """Citation for the specific molecular opacity.


        Returns
        -------
        list of str:
            List of string with reference information

        """
        return []

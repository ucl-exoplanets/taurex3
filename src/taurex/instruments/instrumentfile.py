"""Instrument loaded from file."""
import math
import typing as t

import numpy as np

from taurex.binning import BinDownType, FluxBinner
from taurex.model import ForwardModel
from taurex.types import ModelOutputType, PathLike
from taurex.util import wnwidth_to_wlwidth

from .instrument import Instrument


class InstrumentFile(Instrument):
    """Loads a 2-3 column file

    The first column is the wavelength grid, the second column is the noise
    and the third column is the width of the wavelength bin. If the third column
    is not present, the width is computed from the wavelength grid.


    """

    def __init__(
        self,
        filename: t.Optional[PathLike] = None,
        delimiter: t.Optional[str] = None,
        skiprows: t.Optional[int] = 0,
        use_cols: t.Optional[t.Tuple[int, ...]] = None,
    ) -> None:
        super().__init__()

        self._spectrum = np.loadtxt(
            filename, skiprows=skiprows, delimiter=delimiter, usecols=use_cols
        )

        self._wlgrid = self._spectrum[:, 0]

        sortedwl = self._wlgrid.argsort()[::-1]

        self._wlgrid = self._wlgrid[sortedwl]

        self._wngrid = 10000 / self._wlgrid

        self._noise = self._spectrum[sortedwl, 1]

        try:
            self._wlwidths = self._spectrum[sortedwl, 2]
        except IndexError:
            from taurex.util import compute_bin_edges

            self._wlwidths - compute_bin_edges(self._wlgrid)[-1]

        self.create_wn_widths()

        self._binner = FluxBinner(self._wngrid, wngrid_width=self._wnwidths)

    def create_wn_widths(self) -> None:
        """Covert wavelength widths to wavenumber widths."""
        self._wnwidths = wnwidth_to_wlwidth(self._wlgrid, self._wlwidths)

    def model_noise(
        self,
        model: ForwardModel,
        model_res: t.Optional[ModelOutputType] = None,
        num_observations: t.Optional[int] = 1,
    ) -> BinDownType:
        """Attach noise to forward model.

        Parameters
        ----------

        model:
            Forward model to pass.

        model_res:
            Result from :func:`~taurex.model.model.ForwardModel.model`

        num_observations:
            Number of observations to simulate

        """
        if model_res is None:
            model_res = model.model()

        wngrid, spectrum, _, grid_width = self._binner.bin_model(model_res)

        return wngrid, spectrum, self._noise / math.sqrt(num_observations), grid_width

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        return (
            "file",
            "fromfile",
        )

"""Instrument implementation for SNR noise model."""
import math
import typing as t

import numpy as np

from taurex.binning import BinDownType, Binner
from taurex.model import ForwardModel
from taurex.types import ModelOutputType

from .instrument import Instrument


class SNRInstrument(Instrument):
    """SNR noise model.

    Simple instrument model that, for a given
    wavelength-independant, signal-to-noise ratio,
    compute resulting noise from it.

    """

    def __init__(
        self, SNR: t.Optional[int] = 10, binner: t.Optional[Binner] = None  # noqa: N803
    ):
        """Initialize SNR instrument.

        Parameters
        ----------

        SNR: float
            Signal-to-noise ratio

        binner: :class:`~taurex.binning.binner.Binner`, optional
            Optional resampler to generate a new spectral
            grid.


        """
        super().__init__()

        self._binner = binner
        self._SNR = SNR

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

        binner = self._binner
        if binner is None:
            binner = model.defaultBinner()

        wngrid, spectrum, error, grid_width = self._binner.bin_model(model_res)

        signal = spectrum.max() - spectrum.min()

        noise = np.ones(spectrum.shape) * signal / self._SNR

        return wngrid, spectrum, noise / math.sqrt(num_observations), grid_width

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        return (
            "snr",
            "SNR",
            "signal-noise-ratio",
        )

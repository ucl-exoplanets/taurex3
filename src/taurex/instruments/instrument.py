"""Base instrument model class."""
import typing as t

from taurex.binning import BinDownType
from taurex.data.citation import Citable
from taurex.log import Logger
from taurex.model import ForwardModel
from taurex.types import ModelOutputType


class Instrument(Logger, Citable):
    """Instrument noise model.


    *Abstract class*

    Defines some method that transforms
    a spectrum and generates noise.

    """

    def __init__(self) -> None:
        """Constructor."""
        super().__init__(self.__class__.__name__)

    def model_noise(
        self,
        model: ForwardModel,
        model_res: t.Optional[ModelOutputType] = None,
        num_observations: t.Optional[int] = 1,
    ) -> BinDownType:
        """Model noise for a given forward model.

        **Requires implementation**

        For a given forward model (and optional result)
        Resample the spectrum and compute noise profile.

        Parameters
        ----------

        model: :class:`~taurex.model.model.ForwardModel`
            Forward model to pass.

        model_res: :obj:`tuple`, optional
            Result from :func:`~taurex.model.model.ForwardModel.model`

        num_observations: int, optional
            Number of observations to simulate
        """

        raise NotImplementedError

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Input keywords for instrument."""
        raise NotImplementedError

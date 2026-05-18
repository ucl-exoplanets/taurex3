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

    BIBTEX_ENTRIES = [
        r"""
@ARTICLE{2025A&A...699A.219C,
       author = {{Changeat}, Q. and {Bardet}, D. and {Chubb}, K. and {Dyrek}, A. and {Edwards}, B. and {Ohno}, K. and {Venot}, O.},
        title = "{Cloud and haze parameterization in atmospheric retrievals: Insights from Titan's Cassini data and JWST observations of hot Jupiters}",
      journal = {\aap},
     keywords = {techniques: spectroscopic, planets and satellites: atmospheres, infrared: planetary systems, Earth and Planetary Astrophysics, Instrumentation and Methods for Astrophysics},
         year = 2025,
        month = jul,
       volume = {699},
          eid = {A219},
        pages = {A219},
          doi = {10.1051/0004-6361/202453186},
archivePrefix = {arXiv},
       eprint = {2505.18715},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025A&A...699A.219C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
        """,
    ]

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

# flake8: noqa
from taurex.core import Citable


class TauREXCitations(Citable):
    BIBTEX_ENTRIES = [
        r"""
@ARTICLE{2021ApJ...917...37A,
       author = {{Al-Refaie}, A.~F. and {Changeat}, Q. and {Waldmann}, I.~P. and {Tinetti}, G.},
        title = "{TauREx 3: A Fast, Dynamic, and Extendable Framework for Retrievals}",
      journal = {\apj},
     keywords = {Open source software, Astronomy software, Exoplanet atmospheres, Radiative transfer, Bayesian statistics, Planetary atmospheres, Planetary science, 1866, 1855, 487, 1335, 1900, 1244, 1255, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Earth and Planetary Astrophysics},
         year = 2021,
        month = aug,
       volume = {917},
       number = {1},
          eid = {37},
        pages = {37},
          doi = {10.3847/1538-4357/ac0252},
archivePrefix = {arXiv},
       eprint = {1912.07759},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021ApJ...917...37A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
        """,
        r"""
@ARTICLE{2022ApJ...932..123A,
       author = {{Al-Refaie}, A.~F. and {Changeat}, Q. and {Venot}, O. and {Waldmann}, I.~P. and {Tinetti}, G.},
        title = "{A Comparison of Chemical Models of Exoplanet Atmospheres Enabled by TauREx 3.1}",
      journal = {\apj},
     keywords = {Open source software, Publicly available software, Chemical abundances, Bayesian statistics, Exoplanet atmospheres, Exoplanet astronomy, Exoplanet atmospheric composition, Exoplanets, Radiative transfer, 1866, 1864, 224, 1900, 487, 486, 2021, 498, 1335, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2022,
        month = jun,
       volume = {932},
       number = {2},
          eid = {123},
        pages = {123},
          doi = {10.3847/1538-4357/ac6dcd},
archivePrefix = {arXiv},
       eprint = {2110.01271},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022ApJ...932..123A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}""",
    ]


taurex_citation = TauREXCitations()
__citations__ = taurex_citation.nice_citation()

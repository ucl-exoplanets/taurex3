.. _Citations:

Citation
========

TauREx will output a bibliography at program finish for components used in a run (including plugins) or store a ``.bib``
file when run with ``--bibtex filename.bib``. We also list references for components in the base TauREx3 installation.



Taurex 3
---------
If you use TauREx 3 in your research and publications,
please cite here:

.. code-block:: bibtex

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
    }






Retrieval
---------

If you make use of any of these samplers then please cite the relevant papers

*PyMultiNest* and *MultiNest* ::

    (PyMultiNest)
    X-ray spectral modelling of the AGN obscuring region in the CDFS: Bayesian model selection and catalogue
    J. Buchner, A. Georgakakis, K. Nandra, L. Hsu, C. Rangel, M. Brightman, A. Merloni, M. Salvato, J. Donley and D. Kocevski
    A&A, 564 (2014) A125
    doi: 10.1051/0004-6361/201322971

    MultiNest: an efficient and robust Bayesian inference tool for cosmology and particle physics
    F. Feroz, M.P. Hobson, M. Bridges
    Mon. Not. Roy. Astron. Soc. 398: 1601-1614,2009
    doi: 10.1111/j.1365-2966.2009.14548.x

*PolyChord* ::

    polychord: next-generation nested sampling
    W. J. Handley, M. P. Hobson, A. N. Lasenby
    Mon. Not. Roy. Astron. Soc. 453: 4384–4398,2015
    doi: 10.1093/mnras/stv1911

*dyPolyChord* ::

    Dynamic nested sampling: an improved algorithm for parameter estimation and evidence calculation
    E. Higson, W. Handley, M. Hobson, A. Lasenby
    Statistics and Computing volume 29, 891–913, 2019
    doi: 10.1007/s11222-018-9844-0


.. _preprint: https://arxiv.org/abs/1912.07759

.. _pluginscata:


=================
Plugins Catalogue
=================

.. versionadded:: 3.1

Here is a list of plugins that can be installed to give
TauREx 3 new features and components. Plugins are usually hosted on PyPi
and may have precompiled binary wheels for Windows, MacOS and/or manylinux

.. note::

   Since |version|, the functionality of two previously external plugins has
   been integrated directly into TauREx:

   * **Phoenix4All** — :class:`~taurex.data.stellar.phoenix4all.Phoenix4AllStar`
     replaces the legacy ``PhoenixStar`` (see :doc:`api/taurex.stellar`).
   * **taurex-catalogue** — The :mod:`taurex.catalogue` sub-package provides
     automatic planet/star parameter loading from local files or the ExoMAST
     API (see :doc:`api/taurex.catalogue`).


+------------------+--------------------------------------------------+------+--------+-------+-----------+
| Name             | Description                                      | PyPi | Wheels |       |           |
+==================+==================================================+======+========+=======+===========+
|                  |                                                  |      | Win64  | MacOS | manylinux |
+------------------+--------------------------------------------------+------+--------+-------+-----------+
| acepython        | Equilibrium chemistry using ACE                  | ✔    | ✔      | ✔     | ✔         |
+------------------+--------------------------------------------------+------+--------+-------+-----------+
| taurex_fastchem  | Equilibrium chemistry using FastChem             | ✔    | ✔      | ✔     | ✔         |
+------------------+--------------------------------------------------+------+--------+-------+-----------+
| taurex_ggchem    | Equilibrium chemistry using GGChem               | ✔    | ✔      | ✔     | ✔         |
+------------------+--------------------------------------------------+------+--------+-------+-----------+
| taurex_cuda      | CUDA-acceleration of forward models              | ✔    |        |       |           |
+------------------+--------------------------------------------------+------+--------+-------+-----------+
| taurex_dynesty   | Dynesty optimizer                                | ✔    | ✔      | ✔     | ✔         |
+------------------+--------------------------------------------------+------+--------+-------+-----------+




.. _acepython: http://pypi.org/projects/acepython

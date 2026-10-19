.. _pluginscata:


=================
Plugins Catalogue
=================

.. versionadded:: 3.1

Here is a list of plugins that can be installed to give
TauREx 3 new features and components. Plugins are usually hosted on PyPi
and may have precompiled binary wheels for Windows, MacOS and/or manylinux

.. note::

   Since |version|, the functionality of several previously external plugins has
   been integrated directly into TauREx:

   * **taurex-instrumentsystematics** — the ``spectra_w_offsets`` and
     ``spectra_instr`` observation types (see :ref:`userobservation`).
   * **Phoenix4All** — :class:`~taurex.data.stellar.phoenix4all.Phoenix4AllStar`
     replaces the legacy ``PhoenixStar`` (see :ref:`userstar`).
   * **taurex-catalogue** — The catalogue mixins in :mod:`taurex.mixin` provide
     automatic planet/star parameter loading from local files or the ExoMAST
     API (see :doc:`api/taurex.mixin`).
   * **taurex-PCQ** — the ``PyMieScattGridExtinction`` model contribution (see
     :ref:`usermodel`).
   * **taurex-multimodel** — the composite forward models (see
     :ref:`usermodel`).


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

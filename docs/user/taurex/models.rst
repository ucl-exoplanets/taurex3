.. _usermodel:

===========
``[Model]``
===========

This header defines the type of forward model (FM) that will be computed by TauREx3.
There are several built-in forward ``model_type`` values:
    - ``transmission``
        - Transmission forward model
    - ``emission``
        - Emission forward model
    - ``directimage``
        - Direct-image forward model
    - ``multi_transit``
        - Composite transmission model combining multiple 1D regions
    - ``multi_eclipse``
        - Composite emission model combining multiple 1D regions
    - ``multi_directimage``
        - Composite direct-imaging model combining multiple 1D regions
    - ``custom``
        - User-type forward model, See :ref:`customtypes`

Both emission and direct image also include an optional keyword ``ngauss`` which
dictates the number of Gaussian quadrate points used in the integration. By default
this is set to ``ngauss=4``.

Composite Forward Models
========================

TauREx also includes built-in composite forward models for stitching together
multiple 1D atmospheric regions into a single weighted spectrum. These were
previously distributed through the ``taurex-multimodel`` plugin and are now
available directly in the main package.

The parameter-file-driven entry points are:

+-------------------------+-------------------------------------------------------+
| ``model_type``          | Description                                           |
+-------------------------+-------------------------------------------------------+
| ``multi_transit``       | Weighted combination of multiple transmission regions |
+-------------------------+-------------------------------------------------------+
| ``multi_eclipse``       | Weighted combination of multiple emission regions     |
+-------------------------+-------------------------------------------------------+
| ``multi_directimage``   | Weighted combination of multiple direct-image regions |
+-------------------------+-------------------------------------------------------+

Each region is defined through a separate parameter file listed in ``parfiles``.
TauREx reads the temperature, chemistry, pressure, and contribution sections from
each file and combines the resulting spectra with the optional ``fractions`` list.
For retrievals, provide ``N-1`` fractions rather than ``N`` if you want TauREx to
adapt the last region automatically. In that case the final fraction is inferred
from the remaining weight so that the total remains unity.
For a complete worked setup, see the multimodel notebook example in
:ref:`Examples`.

Example composite transmission setup::

    [Model]
    model_type = multi_transit
    parfiles = day.par, night.par
    fractions = 0.7

The same pattern applies to ``multi_eclipse`` and ``multi_directimage``.

Two internal helper models are also exposed for advanced use:

- ``emission_radscale`` or ``eclipse_radscale`` for radius-scaled emission regions.
- ``direct_radscale`` or ``directimage_radscale`` for radius-scaled direct-image regions.

---------------------------


Contributions
=============

Contributions define what processes in the atmosphere contribute to the optical depth.
These contributions are defined as *subheaders* with the name of the header being the contribution
to add into the forward model.Any forward model type can be augmented with these contributions.


--------
Examples
--------

Transmission spectrum with molecular absorption and CIA from ``H2-He`` and ``H2-H2``::

    [Model]
    model_type = transmission
        [[Absorption]]

        [[CIA]]
        cia_pairs = H2-He,He-He

Emission spectrum with molecular absorption, CIA and Rayleigh scattering::

    [Model]
    model_type = emission
    ngauss = 4
        [[Absorption]]

        [[CIA]]
        cia_pairs = H2-He,He-He

        [[Rayleigh]]

The following sections give a list of available contributions

-----------------------------

Molecular Absorption
====================

``[[Absorption]]``

Adds molecular absorption to the forward model. Here the *active*
molecules contribute to absorption.
No other keywords are needed. No fitting parameters.

---------------------

Collisionally Induced Absorption
================================
``[[CIA]]``

Adds collisionally induced absorption to the forward model.
Requires ``cia_path`` to be set. Both *active* and *inactive*
molecules can contribute.
No fitting parameters

--------
Keywords
--------

+---------------+-------------+------------------------------------------------+
| Variable      | Type        | Description                                    |
+---------------+-------------+------------------------------------------------+
| ``cia_pairs`` | :obj:`list` | List of molecular pairs. e.g. ``H2-He, H2-H2`` |
+---------------+-------------+------------------------------------------------+

---------------------

Rayleigh Scattering
===================
``[[Rayleigh]]``

Adds Rayleigh scattering to the forward model. Both *active* and *inactive*
molecules can contribute. No keywords or fitting parameters.

---------------------

Optically thick clouds
======================
``[[SimpleClouds]]`` or ``[[ThickClouds]]``

A simple cloud model that puts a infinitely absorping cloud deck
in the atmosphere.

--------
Keywords
--------

+---------------------+--------------+-------------------------------------+
| Variable            | Type         | Description                         |
+---------------------+--------------+-------------------------------------+
| ``clouds_pressure`` | :obj:`float` | Pressure of top of cloud-deck in Pa |
+---------------------+--------------+-------------------------------------+

------------------
Fitting Parameters
------------------

+---------------------+--------------+-------------------------------------+
| Variable            | Type         | Description                         |
+---------------------+--------------+-------------------------------------+
| ``clouds_pressure`` | :obj:`float` | Pressure of top of cloud-deck in Pa |
+---------------------+--------------+-------------------------------------+


---------------------------

Mie scattering (Lee)
======================
``[[LeeMie]]``

Computes Mie scattering contribution to optical depth
Formalism taken from: Lee et al. 2013, ApJ, 778, 97

--------
Keywords
--------

+-----------------------+--------------+----------------------------+
| Variable              | Type         | Description                |
+-----------------------+--------------+----------------------------+
| ``lee_mie_radius``    | :obj:`float` | Particle radius in um      |
+-----------------------+--------------+----------------------------+
| ``lee_mie_q``         | :obj:`float` | Extinction coefficient     |
+-----------------------+--------------+----------------------------+
| ``lee_mie_mix_ratio`` | :obj:`float` | Mixing ratio in atmosphere |
+-----------------------+--------------+----------------------------+
| ``lee_mie_bottomP``   | :obj:`float` | Bottom of cloud deck in Pa |
+-----------------------+--------------+----------------------------+
| ``lee_mie_topP``      | :obj:`float` | Top of cloud deck in Pa    |
+-----------------------+--------------+----------------------------+

------------------
Fitting Parameters
------------------

+-----------------------+--------------+----------------------------+
| Parameter             | Type         | Description                |
+-----------------------+--------------+----------------------------+
| ``lee_mie_radius``    | :obj:`float` | Particle radius in um      |
+-----------------------+--------------+----------------------------+
| ``lee_mie_q``         | :obj:`float` | Extinction coefficient     |
+-----------------------+--------------+----------------------------+
| ``lee_mie_mix_ratio`` | :obj:`float` | Mixing ratio in atmosphere |
+-----------------------+--------------+----------------------------+
| ``lee_mie_bottomP``   | :obj:`float` | Bottom of cloud deck in Pa |
+-----------------------+--------------+----------------------------+
| ``lee_mie_topP``      | :obj:`float` | Top of cloud deck in Pa    |
+-----------------------+--------------+----------------------------+

---------------------------

Mie scattering (Precomputed grids)
===================================
``[[PyMieScattGridExtinction]]``

Computes cloud extinction from precomputed :math:`Q_\mathrm{ext}` grids,
promoting the former ``taurex-PCQ`` plugin into the main TauREx codebase.
This is useful when you want PyMieScatt-style cloud retrievals without paying
 the cost of recomputing Mie efficiencies for every model evaluation.

Each species points to an HDF5 grid file containing:

- ``radius_grid`` in microns
- ``wavenumber_grid`` in :math:`cm^{-1}`
- ``Qext`` or ``Qext_grid`` with shape ``(n_radius, n_wavenumber)``

The same contribution works in transmission, emission, and direct-image forward
models because it supplies a wavelength-dependent extinction profile to the
standard TauREx contribution pipeline.

--------
Keywords
--------

+------------------------------------+--------------+--------------------------------------------------------------+
| Variable                           | Type         | Description                                                  |
+------------------------------------+--------------+--------------------------------------------------------------+
| ``species``                        | :obj:`list`  | Names used to label each cloud species and its fit params    |
+------------------------------------+--------------+--------------------------------------------------------------+
| ``mie_species_path``               | :obj:`list`  | Paths to the precomputed aerosol grid files                  |
+------------------------------------+--------------+--------------------------------------------------------------+
| ``mie_particle_radius_distribution`` | :obj:`str` | ``normal``, ``budaj``, or ``deirmendjian`` particle sampling |
+------------------------------------+--------------+--------------------------------------------------------------+
| ``mie_particle_mean_radius``       | :obj:`list`  | Mean particle radius in um                                   |
+------------------------------------+--------------+--------------------------------------------------------------+
| ``mie_particle_logstd_radius``     | :obj:`list`  | Log-normal width used for ``normal`` and sampling control    |
+------------------------------------+--------------+--------------------------------------------------------------+
| ``mie_particle_paramA/B/C/D``      | :obj:`list`  | Shape parameters for the ``deirmendjian`` distribution       |
+------------------------------------+--------------+--------------------------------------------------------------+
| ``mie_particle_radius_Nsampling``  | :obj:`int`   | Number of radius samples used to integrate the distribution  |
+------------------------------------+--------------+--------------------------------------------------------------+
| ``mie_particle_radius_Dsampling``  | :obj:`float` | Width of the sampled radius interval in log space            |
+------------------------------------+--------------+--------------------------------------------------------------+
| ``mie_particle_mix_ratio``         | :obj:`list`  | Particle number density in :math:`m^{-3}`                    |
+------------------------------------+--------------+--------------------------------------------------------------+
| ``mie_midP``                       | :obj:`list`  | Cloud centre pressure in Pa, or ``-1`` for the full column   |
+------------------------------------+--------------+--------------------------------------------------------------+
| ``mie_rangeP``                     | :obj:`list`  | Cloud vertical extent in log-pressure space                  |
+------------------------------------+--------------+--------------------------------------------------------------+
| ``mie_particle_altitude_distrib``  | :obj:`str`   | ``exp_decay`` or ``linear`` vertical particle profile        |
+------------------------------------+--------------+--------------------------------------------------------------+
| ``mie_particle_altitude_decay``    | :obj:`list`  | Decay exponent used by ``exp_decay``                         |
+------------------------------------+--------------+--------------------------------------------------------------+

------------------
Fitting Parameters
------------------

The contribution registers both shared and per-species fitting parameters.
The main ones are:

+----------------------+--------------+--------------------------------------------------+
| Parameter            | Type         | Description                                      |
+----------------------+--------------+--------------------------------------------------+
| ``Rmean_share``      | :obj:`float` | Shared particle radius applied to all species    |
+----------------------+--------------+--------------------------------------------------+
| ``Rlogstd_share``    | :obj:`float` | Shared log-width for supported distributions     |
+----------------------+--------------+--------------------------------------------------+
| ``X_share``          | :obj:`float` | Shared particle number density                   |
+----------------------+--------------+--------------------------------------------------+
| ``midP_share``       | :obj:`float` | Shared cloud centre pressure                     |
+----------------------+--------------+--------------------------------------------------+
| ``rangeP_share``     | :obj:`float` | Shared log-pressure extent                       |
+----------------------+--------------+--------------------------------------------------+
| ``decayP_share``     | :obj:`float` | Shared exponential-decay index                   |
+----------------------+--------------+--------------------------------------------------+
| ``Rmean_<species>``  | :obj:`float` | Per-species particle radius                      |
+----------------------+--------------+--------------------------------------------------+
| ``Rlogstd_<species>``| :obj:`float` | Per-species log-width                            |
+----------------------+--------------+--------------------------------------------------+
| ``X_<species>``      | :obj:`float` | Per-species particle number density              |
+----------------------+--------------+--------------------------------------------------+
| ``midP_<species>``   | :obj:`float` | Per-species cloud centre pressure                |
+----------------------+--------------+--------------------------------------------------+
| ``rangeP_<species>`` | :obj:`float` | Per-species cloud extent                         |
+----------------------+--------------+--------------------------------------------------+
| ``decayP_<species>`` | :obj:`float` | Per-species exponential-decay index              |
+----------------------+--------------+--------------------------------------------------+

Example parameter-file usage::

    [Model]
    model_type = transmission

        [[Absorption]]

        [[PyMieScattGridExtinction]]
        species = Mg2SiO4_glass, SiO2
        mie_species_path = /path/Mg2SiO4_glass.h5, /path/SiO2.h5
        mie_particle_radius_distribution = budaj
        mie_particle_mean_radius = 0.1, 0.5
        mie_particle_mix_ratio = 1e8, 5e7
        mie_midP = 1e5, 1e3
        mie_rangeP = 2.0, 1.0
        mie_particle_altitude_distrib = exp_decay
        mie_particle_altitude_decay = -4.0, -5.0

---------------------------

Mie scattering (BH)
======================
``[[BHMie]]``

Computes a Mie scattering contribution using method given by
Bohren & Huffman 2007

--------
Keywords
--------

+------------------------+-----------------------+----------------------------------------+
| Variable               | Type                  | Description                            |
+------------------------+-----------------------+----------------------------------------+
| ``bh_particle_radius`` | :obj:`float`          | Particle radius in um                  |
+------------------------+-----------------------+----------------------------------------+
| ``bh_cloud_mix``       | :obj:`float`          | Mixing ratio in atmosphere             |
+------------------------+-----------------------+----------------------------------------+
| ``bh_clouds_bottomP``  | :obj:`float`          | Bottom of cloud deck in Pa             |
+------------------------+-----------------------+----------------------------------------+
| ``bh_clouds_topP``     | :obj:`float`          | Top of cloud deck in Pa                |
+------------------------+-----------------------+----------------------------------------+
| ``mie_path``           | :obj:`str`            | Path to molecule scattering parameters |
+------------------------+-----------------------+----------------------------------------+
| ``mie_type``           | ``cloud`` or ``haze`` | Type of mie cloud                      |
+------------------------+-----------------------+----------------------------------------+

------------------
Fitting Parameters
------------------

+------------------------+-----------------------+----------------------------------------+
| Parameter              | Type                  | Description                            |
+------------------------+-----------------------+----------------------------------------+
| ``bh_particle_radius`` | :obj:`float`          | Particle radius in um                  |
+------------------------+-----------------------+----------------------------------------+
| ``bh_cloud_mix``       | :obj:`float`          | Mixing ratio in atmosphere             |
+------------------------+-----------------------+----------------------------------------+
| ``bh_clouds_bottomP``  | :obj:`float`          | Bottom of cloud deck in Pa             |
+------------------------+-----------------------+----------------------------------------+
| ``bh_clouds_topP``     | :obj:`float`          | Top of cloud deck in Pa                |
+------------------------+-----------------------+----------------------------------------+


---------------------------

Mie scattering (Flat)
======================
``[[FlatMie]]``

Computes a flat absorbing region of the atmosphere
across all wavelengths

--------
Keywords
--------

+--------------------+--------------+----------------------------------+
| Variable           | Type         | Description                      |
+--------------------+--------------+----------------------------------+
| ``flat_mix_ratio`` | :obj:`float` | Opacity value                    |
+--------------------+--------------+----------------------------------+
| ``flat_bottomP``   | :obj:`float` | Bottom of absorbing region in Pa |
+--------------------+--------------+----------------------------------+
| ``flat_topP``      | :obj:`float` | Top of absorbing region in Pa    |
+--------------------+--------------+----------------------------------+

------------------
Fitting Parameters
------------------

+--------------------+--------------+----------------------------------+
| Parameter          | Type         | Description                      |
+--------------------+--------------+----------------------------------+
| ``flat_mix_ratio`` | :obj:`float` | Opacity value                    |
+--------------------+--------------+----------------------------------+
| ``flat_bottomP``   | :obj:`float` | Bottom of absorbing region in Pa |
+--------------------+--------------+----------------------------------+
| ``flat_topP``      | :obj:`float` | Top of absorbing region in Pa    |
+--------------------+--------------+----------------------------------+

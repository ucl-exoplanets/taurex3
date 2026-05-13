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

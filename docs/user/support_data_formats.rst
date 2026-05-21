.. _supported_data_formats:

======================
Supported Data Formats
======================


Cross-sections
~~~~~~~~~~~~~~
Supported formats are:
 
- ``.pickle`` *TauREx2* pickle format
- ``.hdf5``, ``.h5`` New HDF5 format
- ``.dat``,  ExoTransmit_ format

More formats can be included through :ref:`plugins`

.. tip::

    For opacities we recommend using hi-res cross-sections (R>7000)
    from a high temperature linelist. Our recommendation are
    linelists from the ExoMol_ project.

K-Tables
~~~~~~~~

.. versionadded:: 3.1


Supported formats are:

- ``.pickle`` *TauREx2* pickle format
- ``.hdf5``, ``.h5`` petitRADTRANS HDF5 format
- ``.kta``,  NEMESIS format

More formats can be included through :ref:`plugins`


Observation
~~~~~~~~~~~

For observations, the following formats supported
are:

- Text based 3/4-column data
- Multi-file text observations with fitted offsets/slopes via ``spectra_w_offsets``
- Multi-file text observations with optional convolution via ``spectra_instr``
- ``.pickle`` Outputs from Iraclis_

More formats can be included through :ref:`plugins`


Cloud Grid Files
~~~~~~~~~~~~~~~~

The built-in ``PyMieScattGridExtinction`` contribution reads precomputed cloud
grids from HDF5 files. Each file must contain:

- ``radius_grid`` with particle radii in microns
- ``wavenumber_grid`` in :math:`cm^{-1}`
- ``Qext`` or ``Qext_grid`` with shape ``(n_radius, n_wavenumber)``

These files are external science data products rather than TauREx-native output
files, but they can now be used directly without installing a separate plugin.


Collisionally Induced Absorption
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only a few formats are supported

- ``.db`` *TauREx2* CIA pickle files
- ``.cia`` HITRAN_ cia files

.. _HITRAN: https://hitran.org/cia/

.. _ExoTransmit: https://github.com/elizakempton/Exo_Transmit/tree/master/Opac

.. _Iraclis: https://github.com/ucl-exoplanets/Iraclis

.. _ExoMol: http://www.exomol.com

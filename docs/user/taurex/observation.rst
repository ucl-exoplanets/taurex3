.. _userobservation:

=================
``[Observation]``
=================

This header deals with loading in spectral data
for retrievals or plotting.

--------
Keywords
--------

Only one of these is required. All accept a string path to a file

+-------------------------+---------------------------------------------------------------------+
| Variable                | Data format                                                         |
+-------------------------+---------------------------------------------------------------------+
| ``observed_spectrum``   | ASCII 3/4-column data with format: Wavelength, depth, error, widths |
+-------------------------+---------------------------------------------------------------------+
| ``observation``         | Observation class keyword for built-in custom loaders               |
+-------------------------+---------------------------------------------------------------------+
| ``observed_lightcurve`` | Lightcurve pickle data                                              |
+-------------------------+---------------------------------------------------------------------+
| ``iraclis_spectrum``    | Iraclis output pickle data                                          |
+-------------------------+---------------------------------------------------------------------+
| ``taurex_spectrum``     | TauREX HDF5 output or ``self`` See taurexspectrum_                  |
+-------------------------+---------------------------------------------------------------------+

-------
Example
-------

An example of loading an ascii data-set::

    [Observation]
    observed_spectrum = /path/to/data.dat


Multi-Instrument Systematics
----------------------------

TauREx also includes built-in observation loaders for combining multiple
ASCII spectra and fitting simple per-instrument systematics directly in the
``[Observation]`` block.

Two observation keywords are available:

+--------------------------+---------------------------------------------------------------+
| Keyword                  | Purpose                                                       |
+--------------------------+---------------------------------------------------------------+
| ``spectra_w_offsets``    | Multiple spectra with per-spectrum offsets, slopes and errors |
+--------------------------+---------------------------------------------------------------+
| ``spectra_instr``        | Same as above, with optional broadening-profile convolution   |
+--------------------------+---------------------------------------------------------------+

These are selected through the generic ``observation`` field::

    [Observation]
    observation = spectra_w_offsets
    path_spectra = /path/spec_1.dat, /path/spec_2.dat
    offsets = 0.0, 0.0
    slopes = 0.0, 0.0
    error_scale = 1.0, 1.0

Each input spectrum should be a 3- or 4-column ASCII file with the usual
TauREx format: wavelength in microns, observed depth, uncertainty, and
optionally bin width.

The following fitting parameters are created automatically for each input
spectrum:

- ``Offset_1``, ``Offset_2``, ...
- ``Slope_1``, ``Slope_2``, ...
- ``EScale_1``, ``EScale_2``, ...

This allows a retrieval configuration such as::

    [Observation]
    observation = spectra_instr
    path_spectra = /path/spec_1.dat, /path/spec_2.dat
    offsets = 0.0, 0.0
    slopes = 0.0, 0.0
    error_scale = 1.0, 1.0
    broadening_profiles = /path/profile_1.fits, /path/profile_2.fits

    [Fitting]
    Offset_1:fit = True
    Offset_2:fit = True
    Slope_1:fit = True
    Slope_2:fit = True

For ``spectra_instr``, the optional ``broadening_profiles`` entries may point
to STScI-style FITS files containing ``WAVELENGTH`` and ``R`` columns, or to
two-column text files containing wavelength and resolving power.


.. _taurexspectrum:

TauREx Spectrum
---------------

The ``taurex_spectrum`` has two different modes. The first mode is specifing a filename path of a
a TauREx3 HDF5 output. This output must have been run with some form of instrument function (see :ref:`userinstrument`),
for it to be useable as an observation.
Another is to set ``taurex_spectrum = self``, this will set the current forward model + instrument function
as the observation. This type observation is valid of the fitting procedure making it possible to do *self-retrievals*.

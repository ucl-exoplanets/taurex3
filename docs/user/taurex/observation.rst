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
``[Observation]`` block. Instrument-response handling now lives on the
active binner, so convolution and wavelength-shift settings can be configured
independently of the observation loader and then reused by any instrument that
consumes that binner.

Two observation keywords are available:

+--------------------------+---------------------------------------------------------------+
| Keyword                  | Purpose                                                       |
+--------------------------+---------------------------------------------------------------+
| ``spectra_w_offsets``    | Multiple spectra with per-spectrum offsets, slopes and errors |
+--------------------------+---------------------------------------------------------------+
| ``spectra_instr``        | Backwards-compatible alias with built-in response binning     |
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
    observation = spectra_w_offsets
    path_spectra = /path/spec_1.dat, /path/spec_2.dat
    offsets = 0.0, 0.0
    slopes = 0.0, 0.0
    error_scale = 1.0, 1.0

    [Binning]
    bin_type = observed
    broadening_type = stsci_fits
    broadening_profiles = /path/profile_1.fits, /path/profile_2.fits
    wlshift = 0.0, 0.0

    [Fitting]
    Offset_1:fit = True
    Offset_2:fit = True
    Slope_1:fit = True
    Slope_2:fit = True

The optional ``broadening_profiles`` entries may point to STScI-style FITS
files containing ``WAVELENGTH`` and ``R`` columns, or to two-column text files
containing wavelength and resolving power. ``wlshift`` accepts either a single
shift applied to every input spectrum or one shift per spectrum.

If you are migrating an older setup, ``observation = spectra_instr`` remains
available and creates the same convolution-aware binner directly from the
``[Observation]`` block.


.. _taurexspectrum:

TauREx Spectrum
---------------

The ``taurex_spectrum`` has two different modes. The first mode is specifing a filename path of a
a TauREx3 HDF5 output. This output must have been run with some form of instrument function (see :ref:`userinstrument`),
for it to be useable as an observation.
Another is to set ``taurex_spectrum = self``, this will set the current forward model + instrument function
as the observation. This type observation is valid of the fitting procedure making it possible to do *self-retrievals*.

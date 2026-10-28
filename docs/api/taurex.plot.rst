Standalone plotting
===================

The plotting functions accept NumPy arrays and an existing Matplotlib axes,
so they can be used in notebooks without an HDF5 output file. Matplotlib must
be installed separately (``pip install matplotlib``).

Pressures are supplied in Pa and displayed in bar. Wavelengths are supplied
in micrometres. The depth array has shape ``(n_layers, n_wavelengths)``;
each coordinate array may be ascending or descending.

.. code-block:: python

   import matplotlib.pyplot as plt
   import taurex.plot as tplot

   # tau, pressure and wavelengths are arrays from your model.
   fig, (ax, profile_ax) = plt.subplots(1, 2, sharey=True)
   mesh = tplot.plot_optical_depth(ax, tau, pressure, wavelengths, cmap="magma")
   fig.colorbar(mesh, ax=ax, label="Optical depth")
   tplot.plot_depth_contrib(profile_ax, tau, pressure, color="black")
   ax.set_title("Optical depth")
   ax.set_xlim(0.5, 12)
   plt.show()

The contribution profile is the arithmetic mean across wavelength samples,
matching the existing Plotter. It is not a wavelength-weighted integral.
Functions return Matplotlib artists for further customization and do not
save or show figures, select a backend, or change global plotting settings.

Temperature–pressure profiles
----------------------------

Temperatures and their optional standard deviations are supplied in kelvin,
with one value per pressure layer. Pressure is supplied in Pa and displayed
in bar, increasing downwards on a logarithmic axis. Either pressure order
is supported, including a single layer. The function returns the line and
the uncertainty band (``None`` when no standard deviations are supplied).

.. code-block:: python

   fig, ax = plt.subplots()
   line, band = tplot.plot_temperature_pressure(
       ax, temperature, pressure, std=temperature_std,
       color="tab:red", label="Retrieved profile",
   )
   ax.legend()
   plt.show()

The band spans temperature plus/minus one standard deviation. The caller
controls titles, legends and saving; additional keyword arguments style
the line. Both forward and fitted TP plots in ``Plotter`` use this function.

Gas abundance profiles
----------------------

Call ``plot_gas_profile`` once per gas, supplying its volume mixing ratios
and matching pressures in Pa. Both axes are logarithmic, with pressure in
bar increasing downwards. The function returns the line and optional band.

.. code-block:: python

   fig, ax = plt.subplots()
   for name, profile, std in zip(gas_names, mixing_ratios, mixing_ratio_stds):
       line, band = tplot.plot_gas_profile(
           ax, profile, pressure, std=std, label=name,
       )
   ax.set_xlim(1e-12, 3)
   ax.legend()
   plt.show()

Arrays contain one value per pressure layer; omit ``std`` for profiles
without uncertainties. Mixing ratios and standard deviations must be finite
and nonnegative. Zero mixing ratios are retained in the line data, but are
not visible on a logarithmic axis.

The uncertainty convention matches the existing retrieval plots: the upper
bound is ``profile + std``, reflected about the profile in logarithmic space
to obtain the lower bound. Equivalently, the lower bound is
``profile**2 / (profile + std)`` for positive profiles, with zero as the limit
at zero mixing ratio. This is not an arithmetic plus/minus interval.
One continuous band replaces the two adjacent legacy bands. Active and
inactive gas plots in ``Plotter`` reuse this function, while retaining their
colors, legends, file names and mixing-ratio limits. Pressure limits follow
the supplied layer range, as in the standalone TP function.

Spectra
-------

``plot_spectrum`` accepts wavelength samples in micrometres, spectral values,
and optional standard deviations in the same units as the spectrum. It
returns the line and a tuple of the one- and two-sigma bands (an empty tuple
without uncertainties). Arrays must be one-dimensional, nonempty and finite;
wavelengths must be positive and standard deviations nonnegative.

.. code-block:: python

   fig, ax = plt.subplots()
   line, bands = tplot.plot_spectrum(
       ax, wavelengths, spectrum, std=spectrum_std,
       color="tab:blue", label="Model",
   )
   ax.set_ylabel("Transit depth")
   ax.set_xscale("log")
   ax.legend()
   plt.show()

The function retains the supplied sample order and includes zero and negative
spectral values. It performs no rebinning or unit conversion and sets only
the wavelength label. The caller controls scales, limits and the ordinate
label, since its meaning depends on the model. Both bands use the line color;
their opacities are 0.5 and 0.2, matching the historical spectrum plots.

``Plotter._generic_plot`` delegates drawing to this function, so forward,
fitted and contribution plots share it. HDF5 dataset selection, optional
``FluxBinner`` rebinning and the historical selection of strictly positive
spectral samples remain in ``Plotter``. If no positive samples remain,
the adapter draws nothing. An explicitly supplied color now styles both
the line and its bands; previously it only styled the bands.

.. automodule:: taurex.plot
   :members:

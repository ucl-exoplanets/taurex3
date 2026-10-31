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
------------------------------

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

Spectral contributions
----------------------

``plot_spectral_contributions`` draws a mapping of names to spectral arrays
on a shared wavelength grid. Names become line labels, and mapping order
determines drawing order and the default color cycle. Each contribution is
drawn independently: no summing, normalization, selection or rebinning is
performed. This supports model contributions that do not add up linearly.

.. code-block:: python

   fig, ax = plt.subplots()
   artists = tplot.plot_spectral_contributions(
       ax, wavelengths,
       {"Absorption-H2O": water_spectrum, "Rayleigh": rayleigh_spectrum},
       std={"Absorption-H2O": water_std},
   )
   ax.set_ylabel("Transit depth")
   ax.legend()
   plt.show()

The optional standard-deviation mapping may cover only some contributions;
unknown names are rejected. Each returned dictionary entry contains the line
and bands from ``plot_spectrum``. All component arrays are validated before
drawing. Shared keyword arguments style all lines; call separately or modify
returned artists for individual styles. An empty mapping draws nothing.

The simple and full contribution methods in ``Plotter`` select HDF5 groups,
prepare each spectrum using the same rebinning and positive-value selection
as regular spectra, then delegate drawing to this API. Simple plots retain
individual absorption components and totals for other contributions; full
plots retain individual components of every contribution. Simple totals now
also receive a contribution-name label, making them visible in the legend.

.. automodule:: taurex.plot
   :members:

Comparing output files
----------------------

The ``*_files`` adapters accept a nonempty sequence of HDF5 paths and optional
matching labels. They read forward outputs or every retrieval solution, close
the files, and return artists without saving or showing figures. Multimodal
retrievals include solution names in their labels.

.. code-block:: python

   files = ["target_a.h5", "target_b.h5"]
   labels = ["Target A", "Target B"]
   fig, ax = plt.subplots()
   tplot.plot_temperature_pressure_files(ax, files, labels=labels)
   ax.legend()

   fig, ax = plt.subplots()
   tplot.plot_gas_profile_files(ax, files, labels=labels, gases=["H2O"])
   ax.legend()

   fig, ax = plt.subplots()
   tplot.plot_spectrum_files(ax, files, labels=labels, observed=True)
   ax.set_ylabel("Transit depth")
   ax.legend()

   fig = tplot.plot_posterior_files(
       files, labels=labels, parameters=["planet_radius", "T"],
   )

Profile limits cover the union of the plotted pressure grids, including when
calling the array functions repeatedly. Set custom limits after plotting.
Uncertainty bands are optional. Spectra use existing binned samples (including
legacy dataset names) or native samples without rebinning or positive-value
filtering. Observations are drawn once per file. Compare spectra with compatible
physical meanings and units; the adapter leaves the ordinate label to you.

Corner comparisons require retrieval outputs. Selected fitting parameters are
aligned by name and ordered as requested, even if stored in different column
orders. By default, all parameters in the first retrieval are selected; missing
parameters raise an error. Derived parameters are not included. Weights are
normalized separately per solution, and default parameter ranges cover all
samples across files and solutions. Thus overlaid posterior shapes compare
each solution independently, without encoding relative solution evidences.
The returned corner figure includes a file/solution legend. Additional keyword
arguments configure the bundled corner plotter, for example ``bins`` or
``range``. Parameter names in the example must match those in your files.

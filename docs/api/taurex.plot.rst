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

.. automodule:: taurex.plot
   :members:

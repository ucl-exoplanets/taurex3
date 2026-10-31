.. _userplotting:

========
Plotting
========

Along with the *--plot* argument for taurex, there
is also an extra program specifically for plotting to PDF
from TauREx 3 HDF5 outputs. It is accessed like this::

    taurex-plot

A summary of the arguments is given here:

+---------------+--------------------+-------------+-----------------------------------+
| Argument      |  Alternate name    | Input       |  Description                      |
+---------------+--------------------+-------------+-----------------------------------+
| -h            |  --help            |             |  show this help message and exit  |
+---------------+--------------------+-------------+-----------------------------------+
| -i            |  --input           | INPUT_FILES |  TauREx 3 HDF5 output files       |
+---------------+--------------------+-------------+-----------------------------------+
| -o            |  --output-dir      | OUTPUT_DIR  |  Directory to store plots         |
+---------------+--------------------+-------------+-----------------------------------+
| -T            |  --title           |   TITLE     |  Title of plots (optional)        |
+---------------+--------------------+-------------+-----------------------------------+
| -p            |  --prefix          |  PREFIX     |  Output filename prefix (optional)|
+---------------+--------------------+-------------+-----------------------------------+
| -m            |  --color-map       |   CMAP      |  Matplotlib colormap (optional)   |
+---------------+--------------------+-------------+-----------------------------------+
| -R            |  --resolution      | RESOLUTION  |  Resample spectra at resolution   |
+---------------+--------------------+-------------+-----------------------------------+
| -P            |  --plot-posteriors |             |  Plot posteriors                  |
+---------------+--------------------+-------------+-----------------------------------+
| -x            |  --plot-xprofile   |             |  Plot molecular profiles          |
+---------------+--------------------+-------------+-----------------------------------+
| -t            |  --plot-tpprofile  |             |  Plot Temperature profiles        |
+---------------+--------------------+-------------+-----------------------------------+
| -d            |  --plot-tau        |             |  Plot optical depth contribution  |
+---------------+--------------------+-------------+-----------------------------------+
| -s            |  --plot-spectrum   |             |  Plot spectrum                    |
+---------------+--------------------+-------------+-----------------------------------+
| -c            |  --plot-contrib    |             |  Plot contrib                     |
+---------------+--------------------+-------------+-----------------------------------+
| -C            |  --full-contrib    |             |  Plot detailed contribs           |
+---------------+--------------------+-------------+-----------------------------------+
| -a            |  --all             |             |  Plot everythiong                 |
+---------------+--------------------+-------------+-----------------------------------+

Comparing multiple output files
-------------------------------

Supply multiple paths after ``-i`` to compare targets or retrievals on shared
axes. With a single path, the existing plotting behavior is retained.
For example, compare temperature profiles, gas abundances and model spectra
with their observations::

    taurex-plot -i target_a.h5 target_b.h5 -o comparison/ \
        --plot-tpprofile --plot-xprofile --plot-spectrum \
        --labels "Target A" "Target B" --prefix targets --title "Comparison"

This writes ``targets_tp_profile.pdf``, ``targets_mixratio.pdf`` and
``targets_spectrum.pdf``. Without ``--prefix``, the prefix is ``output``.
The output directory is created if necessary. Existing files with the same
names are overwritten, as in single-file mode.

``--labels`` is optional and requires one label per input path; defaults are
file stems. All retrieval solutions are included, with solution names added
for files containing multiple solutions. Forward outputs and retrieval outputs
can be compared together for profiles and spectra. Pressure limits cover all
profiles. Gas labels include the species name. ``--color-map`` controls the
plot color cycle, and ``--title`` sets the figure title.

For weighted corner plots, use retrieval files only::

    taurex-plot -i retrieval_a.h5 retrieval_b.h5 -o comparison/ \
        --plot-posteriors --parameters planet_radius T \
        --labels "Retrieval A" "Retrieval B"

This writes ``output_posteriors.pdf``. Replace the parameter names with names
present in your outputs. ``--parameters`` selects and orders shared fitting
parameters; without it, all fitting parameters in the first file are required
in subsequent files. Columns are aligned by parameter name. Derived parameters
are excluded. Each solution's weights are normalized separately, so the plot
compares posterior shapes without representing relative solution evidences.
Default axis ranges include samples from every selected solution.

In multiple-file mode, ``--all`` means the four supported comparisons:
temperature, gas abundances, spectra and posteriors. It therefore requires
retrieval files only. For forward files, select the desired profile and spectrum
flags explicitly. At least one plotting flag is required in multiple-file mode.

Multiple-file mode does not support ``--plot-tau``, ``--plot-cprofile``,
``--plot-contrib``, ``--full-contrib`` or ``--resolution``; these combinations
produce an argument error. Spectra use the stored binned or native samples
without rebinning, and observations appear once per file. Compare spectra with
compatible physical meanings and units; the CLI leaves the spectral ordinate
unlabelled. Use the Python API to customize axes and labels further.
``--labels`` and ``--parameters`` require multiple input files, and
``--parameters`` requires ``--plot-posteriors`` or ``--all``.

See :doc:`../../api/taurex.plot` for the array and HDF5 comparison APIs.

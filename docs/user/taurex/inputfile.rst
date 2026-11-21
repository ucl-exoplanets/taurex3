.. _inputfile:

=================
Input File Format
=================


Sections and values
-------------------

Save TauREx input files with a ``.yaml`` or ``.yml`` extension. Each
top-level key names a section. Indent parameters and subsections beneath
their parent and use ``#`` for comments.

.. code-block:: yaml

    Header:
      parameter1: value
      Subheader:
        parameter2: another value

    Header2:
      integer_value: 10
      float_value: 1.5
      list_value: [hello, how, are, you]

The available sections are:

- :ref:`userglobal`
- :ref:`userchemistry`
- :ref:`usertemperature`
- :ref:`userpressure`
- :ref:`userplanet`
- :ref:`userstar`
- :ref:`usermodel`
- :ref:`userobservation`
- :ref:`userbinning`
- :ref:`userinstrument`
- :ref:`useroptimizer`
- :ref:`userfitting`

Some sections can be omitted to use default profiles. Retrievals require
an observation, an optimizer and the relevant fitting settings.

Dynamic variables
-----------------

The input file is actually a dynamic format and its available variables can change depending
on the choice of certain profiles and types. For example consider the ``Temperature`` section,
it contains the variable ``profile_type`` which describes which temperature profile to use.
Setting this to ``isothermal`` gives us the ``T`` variable which defines the isothermal temperature:

.. code-block:: yaml

    Temperature:
      profile_type: isothermal
      T: 1500.0

Now if we change the profile type to ``guillot2010`` it will use the Guillot 2010 temperature profile
which gives access to the variables ``T_irr``, ``kappa_irr``, ``kappa_v1``, ``kappa_v2``  and ``alpha``
instead:

.. code-block:: yaml

    Temperature:
      profile_type: guillot2010
      T_irr: 1500
      kappa_irr: 0.05
      kappa_v1: 0.05
      kappa_v2: 0.05
      alpha: 0.005

However setting ``T`` will throw an error as it doesn't exist anymore:

.. code-block:: yaml

    Temperature:
      profile_type: guillot2010
      # Error is thrown here
      T: 1500
      kappa_irr: 0.05

This also applies to fitting parameters, profiles provide certain fitting parameters
and changing the model means that these parameters may not exist anymore.

The ``Model`` block can also expand dynamically. A basic single-region setup
might look like this:

.. code-block:: yaml

    Model:
      model_type: transmission

      Absorption: {}

If you switch to the integrated multimodel support, the same section accepts
composite-model keys such as ``parfiles`` and ``fractions``:

.. code-block:: yaml

    Model:
      model_type: multi_transit
      parfiles: [day.yaml, night.yaml]
      fractions: 0.7

In that configuration TauREx reads one regional parameter file per entry in
``parfiles`` and combines the resulting spectra with the supplied weights. If
you provide only ``N-1`` fraction values, the final weight is inferred so that
the total remains unity.

That change also expands the fitting-parameter surface. Composite models expose
shared area-fraction parameters such as ``fr_1`` and region-prefixed component
parameters such as ``m1_T`` whenever profiles are not coupled between regions:

.. code-block:: yaml

    Fitting:
      fr_1:fit: true
      m1_T:fit: true

The same pattern applies to ``model_type: multi_eclipse`` and
``model_type: multi_directimage``.


Complete example
----------------

.. code-block:: yaml

    Temperature:
      profile_type: isothermal
      T: 1500

    Chemistry:
      chemistry_type: taurex
      fill_gases: [H2, He]
      ratio: 0.17
      H2O:
        gas_type: constant
        mix_ratio: 1e-4

    Model:
      model_type: transmission
      Absorption: {}
      CIA:
        cia_pairs: [H2-H2, H2-He]

    Fitting:
      T:fit: true
      T:bounds: [1000, 2000]
      T:prior: 'Uniform(bounds=(1000, 2000))'

Use indentation for subsections, YAML sequences for lists, and ``{}`` for empty
sections such as ``Absorption``. Section and parameter names retain their case.
Scalar values use the existing TauREx conversion rules, including scientific
notation and boolean strings. Quoting a numeric value does not disable this
conversion. Keys such as the molecule name ``NO`` remain strings. Duplicate
keys are rejected.

See :doc:`quickstart` to run the YAML example included in the repository.

For ``multi_transit``, you can instead place the regional configurations in the
same file using ``Model.regions``:

.. code-block:: yaml

    Temperature:
      profile_type: isothermal
      T: 1000
    Chemistry:
      chemistry_type: taurex
    Model:
      model_type: multi_transit
      fractions: [0.5, 0.5]
      regions:
        m1:
          Model:
            Absorption: {}
            Rayleigh: {}
        m2:
          Temperature:
            profile_type: isothermal
            T: 800
          Model:
            Absorption: {}
            Rayleigh: {}

Region keys must be ``m1``, ``m2``, etc., consecutive and in that order. This order
matches ``fractions`` and existing regional fitting names such as
``m1_Rmean_share``. ``regions`` must be non-empty and cannot appear together with
``parfiles``. Other model types do not yet support inline regions.

Each region may contain ``Temperature``, ``Chemistry``, ``Pressure`` and ``Model``.
Planet and star are shared from the main configuration. An omitted regional
profile shares the main profile instance; an explicit profile section creates a
separate object using that section and the class defaults, without merging fields
from the main section. Other regional sections are rejected.

A regional ``Model`` accepts contribution sections only. If supplied, its
contributions replace the entire main contribution list and are instantiated
separately for each region. If omitted or empty, the region falls back to the main
contributions, sharing their instances as with regional files. Nested multimodels
and regional model options are not supported. Relative data paths still resolve
from the working directory.

This support does not introduce explicit shared-object references.

Environment variables
---------------------

TauREx supports ``$NAME`` and ``${NAME}`` in parameter values.
This lets users share a configuration while keeping data paths local:

.. code-block:: yaml

    Global:
      xsec_path: ${TAUREX_DATA}/xsec
      cia_path: ${TAUREX_DATA}/cia

Set the variable in the shell before starting TauREx, for example:

.. code-block:: bash

    export TAUREX_DATA="/data/taurex"
    taurex -i input.yaml --retrieval

Each user can choose a different data directory without editing the shared
configuration. The data itself must already be available there.

Expansion applies to string values and list elements at all nesting levels,
including inline regions and separately loaded regional files. Section names
and parameter keys are not expanded. Variable names use letters, digits and
underscores and cannot begin with a digit.

A referenced variable that is not set raises an error naming the variable and
parameter. An explicitly empty variable expands to an empty string. Use ``$$``
for a literal dollar sign, for example ``$$HOME`` to obtain the text ``$HOME``.
Quotes alone do not disable expansion.

Expansion happens once, after parsing the file and before the existing numeric
and boolean conversions. For example, a variable containing ``1200`` can supply
a temperature. Replacement text is not parsed as YAML or split into list items,
and references inside replacement text are not expanded again. Shell commands,
shell default expressions such as ``${NAME:-default}``, and ``~`` expansion are
not supported. Relative paths remain relative to the working directory.


Mixins
------

.. versionadded:: 3.1

:ref:`mixin` can be applied to any base component through the ``+``
operator:

.. code-block:: yaml

    Temperature:
      profile_type: mixin1+mixin2+base

Where we apply ``mixin1`` and ``mixin2`` to a ``base``.
Including mixins will also include their keywords as well. If ``mixin1``
has the keyword ``param1``, ``mixin2`` has ``param2`` and ``base`` has
``another_param`` then we can define in the input file:

.. code-block:: yaml

    Temperature:
      profile_type: mixin1+mixin2+base
      param1: "Hello"       # From mixin 1
      param2: "World"       # From mixin 2
      another_param: 10.0  # From base

Mixins are evaluated in reverse, the last must be a *non-mixin*
for example if we have a ``doubler`` mixin that doubles temperature profiles
then this is valid:

.. code-block:: yaml

    Temperature:
      profile_type: doubler+isothermal

but this is *not valid*:

.. code-block:: yaml

    Temperature:
      profile_type: isothermal+doubler

Additionally we cannot have more than one base class so this is *invalid*:

.. code-block:: yaml

    Temperature:
      profile_type: doubler+isothermal+guillot

The reverse evaluation means that the first mixin will be *applied last*.
If we have another mixin called ``add50`` which adds 50 K to the profile,
then:

.. code-block:: yaml

    Temperature:
      profile_type: doubler+add50+isothermal
      T: 1000

Will result in a temperature profile of :math:`2100~K`. If we instead do this:

.. code-block:: yaml

    Temperature:
      profile_type: add50+doubler+isothermal
      T: 1000

Then the resultant temperature will be :math:`2050~K`.

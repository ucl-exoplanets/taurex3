.. _userglobal:

============
``[Global]``
============

The global section generally handles settings that affect the whole program.

- ``xsec_path``
    - str or list of str
    - Defines the path(s) that contain molecular cross-sections
    - e.g ``xsec_path = path/to/xsec``

- ``xsec_interpolation``
    - ``exp`` or ``linear``
    - Defines whether to use exponential or linear interpolation for temperature
    - e.g ``xsec_interpolation = exp``

- ``in_memory``
    - ``True`` or ``False``
    - For HDF5 opacities. Determines if streamed from file (False) or loaded into memory (True)
    - Default is ``True``
    - e.g ``in_memory = true``

- ``xsec_float32``
    - ``True`` or ``False``
    - When enabled, molecular cross-section data is converted from ``float64`` to
      ``float32`` upon loading into (shared) memory. This halves memory usage per
      opacity file with negligible impact on retrieval accuracy, at the cost of
      slightly reduced numerical precision.
    - Only takes effect when ``in_memory = True``.
    - Default is ``False`` (keep original ``float64`` precision).
    - e.g ``xsec_float32 = True``

- ``cia_path``
    - str or list of str
    - Defines the path(s) that contain CIA cross-sections
    - e.g ``cia_path = path/to/xsec``

- ``ktable_path``
    - str or list of str
    - Defines the path(s) that contain k-tables
    - e.g ``ktable_path = path/to/ktables``

- ``opacity_method``
    - Either ``xsec`` or ``ktables``
    - Choose whether to use molecular cross-sections or correlated k method.
    - e.g ``opacity_method = ktables``

- ``mpi_use_shared``
    - ``True`` or ``False``
    - Exploit MPI 3.0 shared memory to significantly reduce memory usage per node
    - When running under MPI, will only allocate arrays once in a node rather than each process
    - Works on allocations that use this feature (i.e pickle and HDF5 opacities)
    - e.g ``mpi_use_shared = True``

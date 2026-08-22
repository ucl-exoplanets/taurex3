.. _physical-units:

========================
Physical-unit boundaries
========================

TauREx accepts :class:`astropy.units.Quantity` at public component boundaries
where an input has an unambiguous physical meaning. Quantities are converted
immediately to the component's documented internal unit and stored as plain
numeric values. Unitless input retains its historical TauREx meaning for
backwards compatibility.

This policy applies to constructor arguments and public or fitting setters for
the following component groups:

* planetary, stellar, pressure, and temperature profiles;
* gas abundances, bulk-chemistry ratios, and condensate number densities;
* cloud and Mie parameters;
* spectral grids supplied to binners and forward models.

For example, pressure is stored in Pa, temperature in K, spectral wavenumber
in cm⁻¹, particle number density in m⁻³, and dimensionless quantities as plain
fractions. Incompatible dimensions raise Astropy's ``UnitConversionError``.
The shared :func:`taurex.util.convert_to_unit_value` helper implements this
boundary conversion.

Condensate representations
---------------------------

``condensateMixProfile`` remains a dimensionless representation.
``condensateNumberDensityProfile`` is a separate optional chemistry interface
for non-vapour particle number densities, with shape
``(ncondensates, nlayers)`` and internal unit m⁻³. Chemistry plugins should use
``normalize_condensate_number_density`` when accepting equivalent Quantity
input.

Number density is not inferred from mass density. Converting kg/m³ to m⁻³
requires information such as particle mass or material density and particle
size, which is not part of the base chemistry interface.

Scope boundaries
----------------

Unit conversion is deliberately not performed inside low-level numerical
kernels. Opacity interpolation, radiative-transfer integration, and similar
internal routines continue to consume numeric arrays in their documented
internal units.

The following public-looking inputs also remain outside the generic Quantity
contract because their representation is not unambiguous:

* heterogeneous spectrum and instrument tables, whose columns have different
  units and therefore cannot be represented by one two-dimensional Quantity;
* file-driven legacy light-curve inputs;
* ``PowerGas.beta``, whose effective unit depends on the empirical coefficient
  convention;
* ``PhoenixStar.magnitudeK``, which requires a separate decision about
  logarithmic magnitude units.

Supporting these inputs requires a dedicated file schema or component-specific
API rather than implicit conversion at a numeric boundary.

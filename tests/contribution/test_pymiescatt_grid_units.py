"""Tests for unit-aware precomputed-grid Mie inputs."""

import h5py
import numpy as np
import pytest
from astropy import units as u

from taurex.contributions import PyMieScattGridExtinctionContribution


@pytest.fixture
def grid_paths(tmp_path):
    """Create two minimal extinction grids."""
    paths = []
    for index in range(2):
        path = tmp_path / f"cloud_grid_{index}.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("radius_grid", data=[0.05, 0.1, 0.2])
            handle.create_dataset("wavenumber_grid", data=[1000.0, 2000.0])
            handle.create_dataset(
                "Qext",
                data=np.array([[1.0, 1.5], [1.2, 1.7], [1.5, 2.0]]),
            )
        paths.append(str(path))
    return paths


def make_contribution(grid_paths, **kwargs):
    """Create a two-species contribution for boundary tests."""
    return PyMieScattGridExtinctionContribution(
        species=["Mg2SiO4", "SiO2"],
        mie_species_path=grid_paths,
        **kwargs,
    )


def test_grid_mie_constructor_normalizes_quantity_arrays(grid_paths):
    """Constructor quantities should be converted before broadcasting."""
    contribution = make_contribution(
        grid_paths,
        mie_particle_mean_radius=np.array([100, 200]) * u.nm,
        mie_particle_logstd_radius=5 * u.percent,
        mie_particle_mix_ratio=np.array([2, 3]) / u.cm**3,
        mie_porosity=np.array([10, 20]) * u.percent,
        mie_midP=np.array([0.2, 0.1]) * u.bar,
        mie_rangeP=np.array([1, 2]) * u.dimensionless_unscaled,
        mie_nMedium=125 * u.percent,
        mie_particle_radius_dsampling=200 * u.percent,
        mie_particle_altitude_decay=-2 * u.dimensionless_unscaled,
    )

    np.testing.assert_allclose(contribution._mie_particle_mean_radius, [0.1, 0.2])
    np.testing.assert_allclose(contribution._mie_particle_std_radius, [0.05, 0.05])
    np.testing.assert_allclose(contribution._mie_particle_mix_ratio, [2e6, 3e6])
    np.testing.assert_allclose(contribution._mie_porosity, [0.1, 0.2])
    np.testing.assert_allclose(contribution._mie_midP, [20_000.0, 10_000.0])
    np.testing.assert_allclose(contribution._mie_rangeP, [1.0, 2.0])
    np.testing.assert_allclose(contribution._particle_alt_decay, [-2.0, -2.0])
    assert contribution._mie_nMedium == pytest.approx(1.25)
    assert contribution._dsampling == pytest.approx(2.0)
    assert not isinstance(contribution._mie_particle_mix_ratio[0], u.Quantity)


@pytest.mark.parametrize(
    ("parameter_name", "quantity", "expected"),
    [
        ("Rmean_share", 0.2 * u.mm, 200.0),
        ("Rlogstd_share", 10 * u.percent, 0.1),
        ("X_share", 4 / u.cm**3, 4e6),
        ("midP_share", 0.3 * u.bar, 30_000.0),
        ("rangeP_share", 50 * u.percent, 0.5),
        ("decayP_share", -200 * u.percent, -2.0),
        ("Rmean_Mg2SiO4", 300 * u.nm, 0.3),
        ("Rlogstd_Mg2SiO4", 20 * u.percent, 0.2),
        ("Porosity_Mg2SiO4", 25 * u.percent, 0.25),
        ("X_Mg2SiO4", 5 / u.cm**3, 5e6),
        ("midP_Mg2SiO4", 2 * u.mbar, 200.0),
        ("rangeP_Mg2SiO4", 150 * u.percent, 1.5),
        ("decayP_Mg2SiO4", -300 * u.percent, -3.0),
    ],
)
def test_grid_mie_fitting_setters_normalize_quantities(
    grid_paths, parameter_name, quantity, expected
):
    """Shared and per-species setters should normalize quantities."""
    contribution = make_contribution(grid_paths, mie_porosity=[0.0, 0.0])
    parameter = contribution.fitting_parameters()[parameter_name]

    parameter[3](quantity)

    assert parameter[2]() == pytest.approx(expected)
    assert not isinstance(parameter[2](), u.Quantity)


@pytest.mark.parametrize(
    ("keyword", "quantity"),
    [
        ("mie_particle_mean_radius", 1 * u.s),
        ("mie_particle_logstd_radius", 1 * u.kg),
        ("mie_particle_mix_ratio", 1 * u.kg / u.m**3),
        ("mie_midP", 1 * u.m),
        ("mie_rangeP", 1 * u.K),
    ],
)
def test_grid_mie_rejects_incompatible_constructor_units(grid_paths, keyword, quantity):
    """Physical and dimensionless boundaries should reject wrong dimensions."""
    with pytest.raises(u.UnitConversionError):
        make_contribution(grid_paths, **{keyword: quantity})


def test_grid_mie_preserves_legacy_values(grid_paths):
    """Unitless inputs should retain their documented implicit units."""
    contribution = make_contribution(
        grid_paths,
        mie_particle_mean_radius=[0.1, 0.2],
        mie_particle_mix_ratio=[1e8, 2e8],
        mie_midP=[1e4, -1],
    )

    assert contribution._mie_particle_mean_radius == [0.1, 0.2]
    assert contribution._mie_particle_mix_ratio == [1e8, 2e8]
    assert contribution._mie_midP == [1e4, -1]

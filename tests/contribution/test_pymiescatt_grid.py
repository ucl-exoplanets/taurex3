import numpy as np

import pytest


class DummyModel:
    def __init__(self):
        self.nLayers = 4
        self.pressureProfile = np.array([1e5, 1e4, 1e3, 1e2], dtype=np.float64)


def test_pymiescatt_grid_accepts_qext_grid_dataset(tmp_path):
    import h5py

    from taurex.contributions import PyMieScattGridExtinctionContribution

    grid_path = tmp_path / "cloud_grid.h5"
    with h5py.File(grid_path, "w") as handle:
        handle.create_dataset("radius_grid", data=np.array([0.05, 0.1, 0.2]))
        handle.create_dataset("wavenumber_grid", data=np.array([1000.0, 2000.0, 3000.0]))
        handle.create_dataset(
            "Qext_grid",
            data=np.array(
                [
                    [1.0, 1.5, 2.0],
                    [1.2, 1.7, 2.2],
                    [1.5, 2.0, 2.5],
                ]
            ),
        )

    contribution = PyMieScattGridExtinctionContribution(
        species=["SiO2"],
        mie_species_path=[str(grid_path)],
        mie_particle_mean_radius=[0.1],
        mie_particle_logstd_radius=[0.05],
        mie_particle_mix_ratio=[1e8],
        mie_midP=[1e4],
        mie_rangeP=[2.0],
        mie_particle_altitude_decay=[-2.0],
    )

    components = list(contribution.prepare_each(DummyModel(), np.array([1500.0, 2500.0])))

    assert components[0][0] == "PyMieScattGridExt"
    assert contribution.sigma_xsec.shape == (4, 2)
    assert np.any(contribution.sigma_xsec > 0.0)


def test_pymiescatt_grid_rejects_invalid_distribution(tmp_path):
    import h5py

    from taurex.contributions.pymiescatt_grid import InvalidPyMieScattGridException
    from taurex.contributions import PyMieScattGridExtinctionContribution

    grid_path = tmp_path / "cloud_grid.h5"
    with h5py.File(grid_path, "w") as handle:
        handle.create_dataset("radius_grid", data=np.array([0.05, 0.1]))
        handle.create_dataset("wavenumber_grid", data=np.array([1000.0, 2000.0]))
        handle.create_dataset("Qext", data=np.array([[1.0, 1.5], [1.1, 1.6]]))

    with pytest.raises(InvalidPyMieScattGridException):
        PyMieScattGridExtinctionContribution(
            species=["SiO2"],
            mie_species_path=[str(grid_path)],
            mie_particle_radius_distribution="invalid",
        )
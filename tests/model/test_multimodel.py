"""Test MultiModel."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy import units as u


def test_multitransit_pressure_quantity_inputs():
    """Pressure bounds should accept Quantity values and normalize to Pa."""
    from taurex.model import MultiTransitModel

    model = MultiTransitModel(
        temperature_profiles=[None, None],
        chemistry=[None, None],
        pressure_min=[1e-6 * u.bar, 2e-6 * u.bar],
        pressure_max=[1e3 * u.Pa, 2e3 * u.Pa],
        fractions=[0.5],
    )

    np.testing.assert_allclose(model._pressure_min, [0.1, 0.2])
    np.testing.assert_allclose(model._pressure_max, [1000.0, 2000.0])


def test_multitransit_model_normalizes_wavelength_grid():
    """Composite models pass plain inverse-centimetre grids to submodels."""
    from taurex.model import MultiTransitModel

    model = object.__new__(MultiTransitModel)
    model._sub_models = [MagicMock()]
    model._fractions = [1.0]
    model._active_chems = [None]
    model._inactive_chems = [None]
    model._mus = [None]
    model._temperatures = [None]
    model.autofrac = False
    model.initialize_profiles = MagicMock()
    model._sub_models[0].nativeWavenumberGrid = np.linspace(4000.0, 11000.0, 8)
    model._sub_models[0].model.return_value = (
        np.array([5000.0, 10000.0]),
        np.ones(2),
        np.ones(2),
        None,
    )
    chemistry = model._sub_models[0].chemistry
    chemistry.activeGasMixProfile = np.ones((1, 1))
    chemistry.inactiveGasMixProfile = np.ones((1, 1))
    chemistry.muProfile = np.ones(1)
    model._sub_models[0].temperatureProfile = np.ones(1)

    model.model(np.array([2.0, 1.0]) * u.micron)

    passed_grid = model._sub_models[0].model.call_args.kwargs["wngrid"]
    np.testing.assert_allclose(passed_grid, np.linspace(4000.0, 11000.0, 8))
    assert not isinstance(passed_grid, u.Quantity)


def test_multitransit_autofraction_completion():
    """Test multitransit autofraction completion."""
    from taurex.model import MultiTransitModel

    model = MultiTransitModel(
        temperature_profiles=[None, None],
        chemistry=[None, None],
        fractions=[0.35],
    )

    assert len(model._sub_models) == 2
    assert model.autofrac is True
    assert model._fractions == pytest.approx([0.35, 0.65])


def test_multitransit_invalid_fraction_count():
    """Test multitransit invalid fraction count."""
    from taurex.model import MultiTransitModel

    with pytest.raises(ValueError, match="fractions"):
        MultiTransitModel(
            temperature_profiles=[None, None],
            chemistry=[None, None],
            fractions=[0.2, 0.3, 0.5],
        )


def test_parameter_multitransit_setup_keywords():
    """Test parameter multitransit setup keywords."""
    from taurex.model import MultiParameterTransitModel
    from taurex.pressure import SimplePressureProfile

    pressure = SimplePressureProfile(
        nlayers=50, atm_min_pressure=1e-5, atm_max_pressure=1e5
    )
    model = MultiParameterTransitModel(pressure_profile=pressure, parfiles=[])

    keywords = model.setup_keywords()

    assert len(keywords["temperature_profiles"]) == 1
    assert len(keywords["pressure_profile"]) == 1
    assert keywords["pressure_profile"][0] is pressure
    assert keywords["nlayers"] == [50]

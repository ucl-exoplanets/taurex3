import pytest


def test_multitransit_autofraction_completion():
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
    from taurex.model import MultiTransitModel

    with pytest.raises(ValueError, match="fractions"):
        MultiTransitModel(
            temperature_profiles=[None, None],
            chemistry=[None, None],
            fractions=[0.2, 0.3, 0.5],
        )


def test_parameter_multitransit_setup_keywords():
    from taurex.model import MultiParameterTransitModel
    from taurex.pressure import SimplePressureProfile

    pressure = SimplePressureProfile(nlayers=50, atm_min_pressure=1e-5, atm_max_pressure=1e5)
    model = MultiParameterTransitModel(pressure_profile=pressure, parfiles=[])

    keywords = model.setup_keywords()

    assert len(keywords["temperature_profiles"]) == 1
    assert len(keywords["pressure_profile"]) == 1
    assert keywords["pressure_profile"][0] is pressure
    assert keywords["nlayers"] == [50]
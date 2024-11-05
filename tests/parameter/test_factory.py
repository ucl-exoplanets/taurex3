"""Tests factory behaviour."""

import pytest

from taurex.parameter import factory


def test_get_keywordarg_dict_non_mixin():
    class TestClass:
        def __init__(self, a, b, c=1, d=2):
            pass

    kwargs, has_kvar = factory.get_keywordarg_dict(TestClass)

    assert kwargs == {"a": None, "b": None, "c": 1, "d": 2}


def test_get_keywordarg_dict_mixin():
    from taurex.mixin import Mixin

    class TestClass:
        def __init__(self, a, b, c=1, d=2):
            pass

    class TestMixin(Mixin):
        def __init_mixin__(self, e, f, g=10, h=20):
            pass

    class TestClass2(TestMixin, TestClass):
        pass

    kwargs, has_kvar = factory.get_keywordarg_dict(TestClass2)

    assert kwargs == {
        "a": None,
        "b": None,
        "c": 1,
        "d": 2,
        "e": None,
        "f": None,
        "g": 10,
        "h": 20,
    }


def test_create_klass():
    """Test creating a class from a factory."""
    from taurex.temperature import Isothermal

    tp = factory.create_klass({"T": 1000}, Isothermal)

    assert isinstance(tp, Isothermal)
    assert tp.isoTemperature == 1000

    tp = factory.create_klass({"T": 2000}, Isothermal)

    assert isinstance(tp, Isothermal)
    assert tp.isoTemperature == 2000

    with pytest.raises(KeyError):
        factory.create_klass({"T": 2000, "P": 100}, Isothermal)


def test_generic_factory():
    """Test finding best class from factory."""
    from taurex.temperature import Guillot2010, Isothermal, TemperatureProfile

    klass = factory.generic_factory("isothermal", TemperatureProfile)

    assert klass == Isothermal

    klass = factory.generic_factory("guillot", TemperatureProfile)

    assert klass == Guillot2010

    with pytest.raises(KeyError):
        factory.generic_factory("nonexistent", TemperatureProfile)


def test_determine_klass_type():
    """Test whether the factory can determine the correct class."""
    from taurex.temperature import Guillot2010, Isothermal, TemperatureProfile

    klass = factory.determine_klass(
        {
            "type": "isothermal",
        },
        TemperatureProfile,
    )

    assert klass == ({}, Isothermal, False)

    klass = factory.determine_klass(
        {
            "profile_type": "guillot",
            "T_irr": 1000,
        },
        TemperatureProfile,
    )

    assert klass == ({"T_irr": 1000}, Guillot2010, False)


def test_determine_klass_exception():
    """Tests errors in determining the class."""
    from taurex.temperature import TemperatureProfile

    with pytest.raises(KeyError):
        factory.determine_klass(
            {
                "type": "nonexistent",
            },
            TemperatureProfile,
        )
    with pytest.raises(ValueError):
        factory.determine_klass(
            {
                "type": "isothermal",
                "profile_type": "guillot",
            },
            TemperatureProfile,
        )

    with pytest.raises(ValueError):
        factory.determine_klass(
            {},
            TemperatureProfile,
        )


def test_determine_klass_alttype():
    """Test whether the factory can determine the correct class."""
    from taurex.temperature import Guillot2010, Isothermal, TemperatureProfile

    klass = factory.determine_klass(
        {
            "temperature_type": "isothermal",
        },
        TemperatureProfile,
        alt_type="temperature_type",
    )

    assert klass == ({}, Isothermal, False)

    klass = factory.determine_klass(
        {
            "cool_type": "guillot",
        },
        TemperatureProfile,
        alt_type="cool_type",
    )

    assert klass == ({}, Guillot2010, False)


def test_determine_klass_defaulttype():
    """Tests whether the factory can determine the correct class without types."""
    from taurex.temperature import Isothermal, TemperatureProfile

    klass = factory.determine_klass(
        {},
        TemperatureProfile,
        default_type="isothermal",
    )

    assert klass == ({}, Isothermal, False)


def test_determine_klass_mixin():
    """Tests whether the factory can determine the correct class with mixins."""
    from taurex.mixin import TempScaler
    from taurex.temperature import Isothermal, TemperatureProfile

    klass = factory.determine_klass(
        {
            "type": "tempscalar+isothermal",
        },
        TemperatureProfile,
    )

    class_var = klass[1]
    assert klass[-1] is True
    assert issubclass(class_var, TempScaler)
    assert issubclass(class_var, Isothermal)

    # Test with arrays

    klass = factory.determine_klass(
        {
            "type": ["tempscalar", "isothermal"],
        },
        TemperatureProfile,
    )

    class_var = klass[1]
    assert klass[-1] is True
    assert issubclass(class_var, TempScaler)
    assert issubclass(class_var, Isothermal)


def test_create_generic():
    """Test if it can create from generic config."""
    from taurex.optimizer import NestleOptimizer, Optimizer
    from taurex.temperature import Isothermal, TemperatureProfile

    config = {
        "type": "isothermal",
        "T": 1000,
    }

    tp = factory.create_generic(config, TemperatureProfile)

    assert isinstance(tp, Isothermal)
    assert tp.isoTemperature == 1000

    config = {
        "type": "nestle",
        "num_live_points": 1000,
    }

    opt = factory.create_generic(config, Optimizer)

    assert isinstance(opt, NestleOptimizer)

    assert opt.numLivePoints == 1000


def test_create_priors():
    """Test prior creation."""
    from taurex.core.priors import Uniform

    prior = factory.create_prior("Uniform(bounds=[10,40])")

    assert isinstance(prior, Uniform)

    assert prior.boundaries() == (10, 40)


def test_creates():
    """Test all create functions."""
    planet = factory.create_planet(
        {"type": "simple", "planet_radius": 1.0, "planet_mass": 1.0}
    )

    assert planet.radius == 1.0
    assert planet.mass == 1.0

    star = factory.create_star({"type": "blackbody", "temperature": 1000})

    assert star.temperature == 1000

    chem = factory.create_chemistry(
        {"type": "free", "CO": {"type": "constant"}, "CH4": {"type": "constant"}}
    )

    assert chem.gases == ["H2", "He", "CO", "CH4"]

    press = factory.create_pressure_profile(
        {
            "type": "hydrostatic",
            "nlayers": 100,
            "atm_min_pressure": 1e-4,
            "atm_max_pressure": 1e6,
        }
    )

    assert press.nLayers == 100
    assert press._atm_min_pressure == 1e-4
    assert press._atm_max_pressure == 1e6

    temp = factory.create_temperature_profile({"type": "isothermal", "T": 1000})

    assert temp.isoTemperature == 1000

    opt = factory.create_optimizer({"type": "nestle", "num_live_points": 1000})

    assert opt.numLivePoints == 1000

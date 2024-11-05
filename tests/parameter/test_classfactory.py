import typing as t

from taurex.parameter.classfactory import ClassFactory
from taurex.temperature import TemperatureProfile


class TempTest(TemperatureProfile):
    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        return ("test",)


def gen_fake_module():
    """Generates a fake module with a test class."""
    from types import ModuleType

    m = ModuleType("test")

    m.TempTest = TempTest

    return m


def test_load_plugin():
    """Test whether the class factory can load a plugin."""
    cf = ClassFactory()

    fake_module = gen_fake_module()

    cf.load_plugin(fake_module)

    assert TempTest in cf.temperatureKlasses


def test_model_detection():
    """Test whether the class factory can detect models."""
    from taurex.model import DirectImageModel, EmissionModel, TransmissionModel

    cf = ClassFactory()

    assert TransmissionModel in cf.modelKlasses
    assert EmissionModel in cf.modelKlasses
    assert DirectImageModel in cf.modelKlasses


def test_klass_from_base():
    """Test whether the class factory can find classes from base."""
    from taurex.temperature import Guillot2010, Isothermal, TemperatureProfile

    cf = ClassFactory()

    klasses = cf.klass_from_base(TemperatureProfile)

    assert Isothermal in klasses
    assert Guillot2010 in klasses


def test_find_class_from_name():
    """Test whether the class factory can find classes from name."""
    from taurex.optimizer import NestleOptimizer

    cf = ClassFactory()

    klass = cf.find_klass("NestleOptimizer")

    assert klass == NestleOptimizer


def test_find_from_keyword():
    """Test whether the class factory can find classes from base."""
    from taurex.optimizer import NestleOptimizer
    from taurex.temperature import Guillot2010, Isothermal, TemperatureProfile

    cf = ClassFactory()
    # with base
    klass = cf.find_klass_from_keyword("isothermal", TemperatureProfile)

    assert klass == Isothermal
    # without base
    klass = cf.find_klass_from_keyword("guillot")

    assert klass == Guillot2010

    klass = cf.find_klass_from_keyword("nestle")

    assert klass == NestleOptimizer

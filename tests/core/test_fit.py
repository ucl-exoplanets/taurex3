"""Test fitting functionality."""


def test_fitparam_decorator():
    """Test fitparam decorator."""
    from taurex.core import Fittable, fitparam

    class MyClass(Fittable):
        def __init__(self):
            super().__init__()
            self._test = None

        @fitparam(param_name="test")
        def test(self):
            return self._test

        @test.setter
        def test(self, value):
            self._test = value

    mc = MyClass()

    assert mc.test is None

    mc.test = 10

    assert mc.test == 10

    assert "test" in mc.fitting_parameters()

    assert mc.fitting_parameters()["test"][2]() == 10

    mc.fitting_parameters()["test"][3](30)

    assert mc.test == 30

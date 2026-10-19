"""Test core mixin functions."""

from taurex.mixin import core


def test_determine_mixin_args_no_kwargs():
    """Test determine mixin args."""
    from taurex.mixin import Mixin

    class TestClass:
        """Test class."""

        def __init__(self, a, b, c=1, d=2):
            pass

    class TestMixin(Mixin):
        """Test mixin."""

        def __init_mixin__(self, e, f, g=10, h=20):
            pass

    class TestClass2(TestMixin, TestClass):
        """Test class 2."""

        pass

    kwargs, has_kwarg = core.determine_mixin_args(TestClass2.__bases__)

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

    assert has_kwarg is False


def test_determine_mixin_args_with_kwargs():
    """Test determine mixin args with kwargs."""
    from taurex.mixin import Mixin

    class TestClass:
        """Test class."""

        def __init__(self, a, b, c=1, d=2):
            pass

    class TestMixin(Mixin):
        """Test mixin."""

        def __init_mixin__(self, e, f, g=10, h=20, **kwargs):
            pass

    class TestClass2(TestMixin, TestClass):
        """Test class 2."""

        pass

    kwargs, has_kwarg = core.determine_mixin_args(TestClass2.__bases__)

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

    assert has_kwarg is True


def test_optimizer_mixin_fitting_parameters():
    """Check OptimizerMixin does not shadow Optimizer's parameter properties.

    ``Fittable`` (a base of ``Mixin``) defines ``fitting_parameters`` and
    ``derived_parameters`` as methods, which come earlier in the mixed-class
    MRO than ``Optimizer``'s properties of the same name. Regression test for
    https://github.com/ucl-exoplanets/taurex3/issues/65
    """
    from taurex.mixin import OptimizerMixin
    from taurex.mixin import enhance_class
    from taurex.optimizer import Optimizer
    from taurex.optimizer.optimizer import DerivedParam
    from taurex.optimizer.optimizer import FitParam

    from ..optimizer import LineModel
    from ..optimizer import LineObs

    class DoubleLogLikelihood(OptimizerMixin):
        """Mixin that doubles the log-likelihood."""

        def __init_mixin__(self, my_args="Hello"):
            self.my_args = my_args

        def log_likelihood(self, parameters):
            """Double the base log-likelihood."""
            old_log_likelihood = super().log_likelihood(parameters)
            return old_log_likelihood * 2.0

        @classmethod
        def input_keywords(cls):
            """Input keywords for mixin."""
            return ["doublelog"]

    lm = LineModel()
    lm.m = 0.5
    lm.c = 10.0
    lo = LineObs(m=0.5, c=10.0, N=10)

    opt = enhance_class(
        Optimizer,
        [DoubleLogLikelihood],
        name="test",
        observed=lo,
        model=lm,
        my_args="World",
    )

    opt.enable_fit("m")
    opt.enable_derived("mplusc")

    assert isinstance(opt.fitting_parameters, list)
    assert all(isinstance(p, FitParam) for p in opt.fitting_parameters)
    assert [p.name for p in opt.fitting_parameters] == ["m"]

    assert isinstance(opt.derived_parameters, list)
    assert all(isinstance(p, DerivedParam) for p in opt.derived_parameters)
    assert [p.name for p in opt.derived_parameters] == ["mplusc"]

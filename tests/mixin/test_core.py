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

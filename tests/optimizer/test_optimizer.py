"""Test optimizer base class."""

import math
from types import SimpleNamespace

import pytest
from hypothesis import given
from hypothesis import note
from hypothesis import strategies as st

from taurex.core.priors import LogUniform
from taurex.core.priors import Uniform
from taurex.optimizer import Optimizer

from . import LineModel
from . import LineObs
from . import LineObsWithParams


@pytest.mark.parametrize(("rank", "has_solution_products"), [(0, True), (1, False)])
def test_generate_solution_only_builds_forward_products_on_master(
    monkeypatch, rank, has_solution_products
):
    """Only rank 0 builds MAP and median products; all ranks compute uncertainty."""
    import taurex.mpi
    import taurex.util.output

    monkeypatch.setattr(taurex.mpi, "get_rank", lambda: rank)
    monkeypatch.setattr(
        taurex.util.output, "store_contributions", lambda *args, **kwargs: {}
    )

    class Model:
        def __init__(self):
            self.forward_evaluations = 0
            self.profile_generations = 0

        def model(self, **kwargs):
            self.forward_evaluations += 1
            return "spectrum"

        def generate_profiles(self):
            self.profile_generations += 1
            return {"temperature": {"median": 1000.0}}

    class Binner:
        def generate_spectrum_output(self, result, output_size):
            return {"spectrum": result}

    class FakeOptimizer:
        def __init__(self):
            self._model = Model()
            self._binner = Binner()
            self._observed = SimpleNamespace(wavenumberGrid=[1.0])
            self.updated_values = []
            self.uncertainty_calls = 0

        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def get_solution(self):
            yield (0, [1.0], [2.0], [("metadata", "value")])

        def update_model(self, values):
            self.updated_values.append(values)

        def generate_profiles(self, solution, wavenumber_grid):
            self.uncertainty_calls += 1
            return ({"temperature": {"uncertainty": 10.0}}, {})

        @property
        def derived_names(self):
            return []

    fake = FakeOptimizer()
    result = Optimizer.generate_solution(fake)

    assert fake.uncertainty_calls == 1
    if has_solution_products:
        assert fake._model.forward_evaluations == 2
        assert fake._model.profile_generations == 1
        assert fake.updated_values == [[1.0], [2.0]]
        assert result["solution0"]["Profiles"]["temperature"]["uncertainty"] == 10.0
    else:
        assert fake._model.forward_evaluations == 0
        assert fake._model.profile_generations == 0
        assert fake.updated_values == []
        assert result == {}


@given(m=st.floats(0.1, 100, allow_nan=False), c=st.floats(0.1, 100, allow_nan=False))
def test_optimizer_fittingparams(m, c):
    """Test optimizer fitting params."""
    lm = LineModel()
    lm.m = m
    lm.c = c
    lo = LineObs(m=m, c=c, N=10)

    opt = Optimizer("test", observed=lo, model=lm)

    opt.enable_fit("m")

    assert "m" in opt.fit_names
    assert "c" not in opt.fit_names
    assert opt.fit_values[opt.fit_names.index("m")] == m

    opt.enable_fit("c")
    opt.disable_fit("m")

    assert "c" in opt.fit_names
    assert "m" not in opt.fit_names
    assert opt.fit_values[opt.fit_names.index("c")] == c

    opt.disable_fit("m")
    opt.disable_fit("c")

    opt.enable_fit("c")
    opt.enable_fit("m")

    assert "c" in opt.fit_names
    assert "m" in opt.fit_names
    assert opt.fit_values[opt.fit_names.index("c")] == c
    assert opt.fit_values[opt.fit_names.index("m")] == m


@given(m=st.floats(0.1, 100, allow_nan=False), c=st.floats(0.1, 100, allow_nan=False))
def test_optimizer_fittingparams_with_obs(m, c):
    """Test optimizer fitting params with observation."""
    lm = LineModel()
    lm.m = m
    lm.c = c
    lo = LineObsWithParams(m=m, c=c, N=10)

    opt = Optimizer("test", observed=lo, model=lm)

    opt.enable_fit("m")
    opt.enable_fit("lol")

    assert "m" in opt.fit_names
    assert "lol" in opt.fit_names
    assert "c" not in opt.fit_names
    assert opt.fit_values[opt.fit_names.index("m")] == m

    opt.enable_fit("c")
    opt.disable_fit("m")

    assert "c" in opt.fit_names
    assert "m" not in opt.fit_names
    assert "lol" in opt.fit_names
    assert opt.fit_values[opt.fit_names.index("c")] == c

    opt.enable_fit("c")
    opt.disable_fit("m")
    opt.disable_fit("lol")

    assert "c" in opt.fit_names
    assert "m" not in opt.fit_names
    assert "lol" not in opt.fit_names
    assert opt.fit_values[opt.fit_names.index("c")] == c

    opt.disable_fit("m")
    opt.disable_fit("c")
    opt.disable_fit("lol")

    opt.enable_fit("c")
    opt.enable_fit("m")
    opt.enable_fit("lol")

    assert "c" in opt.fit_names
    assert "m" in opt.fit_names
    assert "lol" in opt.fit_names
    assert opt.fit_values[opt.fit_names.index("c")] == c
    assert opt.fit_values[opt.fit_names.index("m")] == m


def test_optimizer_setprior():
    """Test optimizer set prior."""
    from taurex.core.priors import Gaussian
    from taurex.core.priors import Uniform

    lm = LineModel()
    lm.m = 1.0
    lm.c = 10.0
    lo = LineObs(m=1.0, c=10.0, N=10)

    opt = Optimizer("test", observed=lo, model=lm)

    opt.set_prior("m", Uniform([0.1, 100]))
    opt.set_prior("c", Gaussian(0.1, 100))

    opt.enable_fit("m")
    opt.enable_fit("c")

    assert isinstance(opt.fitting_priors[opt.fit_names.index("m")], Uniform)
    assert isinstance(opt.fitting_priors[opt.fit_names.index("c")], Gaussian)


@given(m=st.floats(0.1, 100, allow_nan=False), c=st.floats(0.1, 100, allow_nan=False))
def test_optimizer_updatemodel(m, c):
    """Test optimizer update model."""
    lm = LineModel()
    lm.m = 1.0
    lm.c = 10.0
    lo = LineObs(m=m, c=c, N=10)

    opt = Optimizer("test", observed=lo, model=lm)

    # Nothing test

    with pytest.raises(ValueError):
        opt.update_model([1.0, 10.0])

    # Test each value individually
    opt.enable_fit("m")

    opt.update_model([m])
    assert lm.m == m
    assert lm.c == 10.0

    lm.m = 1.0

    opt.disable_fit("m")
    opt.enable_fit("c")

    opt.update_model([c])
    assert lm.m == 1.0
    assert lm.c == c
    lm.m = 1.0
    lm.c = 10.0

    opt.disable_fit("c")

    # Test order
    opt.enable_fit("m")
    opt.enable_fit("c")

    values = [0, 0]
    values[opt.fit_names.index("m")] = m
    values[opt.fit_names.index("c")] = c

    opt.update_model(values)
    assert lm.c == c
    assert lm.m == m

    lm.c = 10.0
    lm.m = 1.0
    opt.disable_fit("m")
    opt.disable_fit("c")

    opt.enable_fit("c")
    opt.enable_fit("m")

    values = [0, 0]
    values[opt.fit_names.index("m")] = m
    values[opt.fit_names.index("c")] = c

    opt.update_model(values)

    assert lm.c == c
    assert lm.m == m


@given(
    bounds=st.lists(
        st.lists(st.floats(0.01, 100, allow_nan=False), min_size=2, max_size=2),
        min_size=2,
        max_size=2,
    ),
    dolog=st.lists(st.booleans(), min_size=2, max_size=2),
)
def test_optimizer_olduniform(bounds, dolog):
    """Test optimizer old uniform."""
    lm = LineModel()
    lm.m = 1.0
    lm.c = 10.0
    lo = LineObs(m=1, c=1, N=10)
    opt = Optimizer("test", observed=lo, model=lm)
    opt.enable_fit("m")
    opt.enable_fit("c")

    for p, b, l in zip(["m", "c"], bounds, dolog, strict=False):
        if l:
            opt.set_mode(p, "log")
        else:
            opt.set_mode(p, "linear")
        opt.set_boundary(p, b)

    note(opt.fit_names)

    for p, b, l in zip(["m", "c"], bounds, dolog, strict=False):
        param = p
        if l:
            param = f"log_{p}"
        prior = opt.fitting_priors[opt.fit_names.index(param)]
        if l:
            assert isinstance(prior, LogUniform)
            lb = [math.log10(x) for x in b]
            note(lb)
            assert prior._low_bounds == min(*lb)
            assert prior._up_bounds == max(*lb)
        else:
            assert isinstance(prior, Uniform)
            assert prior._low_bounds == min(*b)
            assert prior._up_bounds == max(*b)


@given(m=st.floats(0.1, 100, allow_nan=False), c=st.floats(0.1, 100, allow_nan=False))
def test_optimizer_derived(m, c):
    """Test optimizer derived."""
    lm = LineModel()
    lm.m = m
    lm.c = c
    lo = LineObs(m=m, c=c, N=10)

    opt = Optimizer("test", observed=lo, model=lm)

    opt.enable_derived("mplusc")
    opt.enable_fit("m")

    assert "mplusc" in opt.derived_names
    assert opt.derived_values[opt.derived_names.index("mplusc")] == m + c
    assert "mplusc_derived" not in opt.fit_names

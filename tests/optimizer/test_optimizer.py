import math

import pytest
from hypothesis import given, note
from hypothesis import strategies as st

from taurex.core.priors import LogUniform, Uniform
from taurex.optimizer import Optimizer

from . import LineModel, LineObs, LineObsWithParams


@given(m=st.floats(0.1, 100, allow_nan=False), c=st.floats(0.1, 100, allow_nan=False))
def test_optimizer_fittingparams(m, c):
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
    from taurex.core.priors import Gaussian, Uniform

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
    lm = LineModel()
    lm.m = 1.0
    lm.c = 10.0
    lo = LineObs(m=1, c=1, N=10)
    opt = Optimizer("test", observed=lo, model=lm)
    opt.enable_fit("m")
    opt.enable_fit("c")

    for p, b, l in zip(["m", "c"], bounds, dolog):
        if l:
            opt.set_mode(p, "log")
        else:
            opt.set_mode(p, "linear")
        opt.set_boundary(p, b)

    note(opt.fit_names)

    for p, b, l in zip(["m", "c"], bounds, dolog):
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

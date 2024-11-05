"""Test math"""

import hypothesis
import hypothesis.extra.numpy as hnum
import numpy as np
import pytest
from hypothesis.strategies import floats


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ([0.0, 1.0, 0.5, 0.0, 1.0], 0.5),
        ([0.0, 1.0, 500, 0, 1000], 0.5),
    ],
)
def test_lin(test_input, expected):
    """Test linear interpolation"""
    from taurex.util.math import interp_lin_only

    x11, x12, press, p_min, p_max = test_input
    val = interp_lin_only(np.array([x11]), np.array([x12]), *test_input[2:])

    assert val == expected


@hypothesis.given(
    temp=floats(0, 1),
    press=floats(0, 1),
    x11=floats(1e-30, 1e-1),
    x12=floats(1e-30, 1e-1),
    x21=floats(1e-30, 1e-1),
    x22=floats(1e-30, 1e-1),
)
@hypothesis.settings(deadline=None)  # This requires a compilation stage initially
def test_bilin(temp, press, x11, x12, x21, x22):
    """Test bilinear interpolation"""
    from taurex.util.math import intepr_bilin, interp_lin_only

    p_min, p_max = 0.0, 1.0
    t_min, t_max = 0.0, 1.0
    val = intepr_bilin(
        np.array([x11]),
        np.array([x12]),
        np.array([x21]),
        np.array([x22]),
        temp,
        t_min,
        t_max,
        press,
        p_min,
        p_max,
    )
    assert pytest.approx(val[0]) == interp_lin_only(
        interp_lin_only(np.array([x11]), np.array([x12]), temp, t_min, t_max),
        interp_lin_only(np.array([x21]), np.array([x22]), temp, t_min, t_max),
        press,
        p_min,
        p_max,
    )


@hypothesis.settings(deadline=2000)
@hypothesis.given(
    hnum.arrays(np.float64, hnum.array_shapes(), elements=floats(0.0, 1000))
)
@hypothesis.example(np.array([[0.0, 0.0]]))
def test_online_variance(s):
    from taurex.util.math import OnlineVariance

    num_values = s.shape[0]
    expected = np.std(s, axis=0)

    onv = OnlineVariance()
    for x in s:
        onv.update(x)

    p_var = onv.parallelVariance()
    var = onv.variance
    if num_values < 2:
        assert np.isnan(var)
        assert np.isnan(p_var)
    else:
        assert np.all(np.isclose(var, p_var))
        assert np.sqrt(var) == pytest.approx(expected, rel=1e-6)
        assert np.sqrt(p_var) == pytest.approx(expected, rel=1e-6)
    # onv = Onli

"""Test Nestle optimizer."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from . import BinnedLineObs
from . import LineModel
from . import LineObs
from . import NativeLineModel


@pytest.mark.slow
@given(m=st.floats(1.0, 2.0), c=st.floats(1.0, 30.0))
@settings(deadline=None)
def test_optimizer(m, c):
    """Test optimizer."""
    from taurex.optimizer import NestleOptimizer

    lm = LineModel()
    lm.m = 1.0
    lm.c = 10.0
    lo = LineObs(m=m, c=c, N=5)
    opt = NestleOptimizer(num_live_points=5, observed=lo, model=lm)
    opt.enable_fit("m")
    opt.enable_fit("c")
    opt.enable_derived("mplusc")
    opt.set_boundary("m", [0.8 * m, 1.2 * m])
    opt.set_boundary("c", [0.8 * c, 1.2 * c])

    opt.fit()

    idx, optimized_map, optimized_median, values = next(opt.get_solution())

    opt.update_model(optimized_map)

    assert lm.m == pytest.approx(m, rel=0.2)
    assert lm.c == pytest.approx(c, rel=0.2)


@pytest.mark.slow
def test_optimizer_fluxbinner():
    """Test retrieval with FluxBinner binning.

    Verifies that the flux-conserving binner works correctly when used
    in the retrieval pipeline: the model is computed on its native fine
    grid, then binned to the observation's coarse grid via FluxBinner.
    """
    np.random.seed(42)
    from taurex.optimizer import NestleOptimizer

    true_m, true_c = 1.5, 15.0

    lm = NativeLineModel()
    lm.m = 1.0
    lm.c = 10.0

    lo = BinnedLineObs(m=true_m, c=true_c, N=10)

    opt = NestleOptimizer(num_live_points=50, observed=lo, model=lm)
    opt.enable_fit("m")
    opt.enable_fit("c")
    opt.set_boundary("m", [0.8 * true_m, 1.2 * true_m])
    opt.set_boundary("c", [0.8 * true_c, 1.2 * true_c])

    opt.fit()

    idx, optimized_map, optimized_median, values = next(opt.get_solution())
    opt.update_model(optimized_map)

    assert lm.m == pytest.approx(true_m, rel=0.2)
    assert lm.c == pytest.approx(true_c, rel=0.2)


@pytest.mark.slow
def test_optimizer_simplebinner():
    """Test retrieval with SimpleBinner binning.

    Verifies that the simple histogram-based binner works correctly
    when used in the retrieval pipeline.
    """
    np.random.seed(42)
    from taurex.binning import SimpleBinner
    from taurex.optimizer import NestleOptimizer

    true_m, true_c = 1.5, 15.0

    lm = NativeLineModel()
    lm.m = 1.0
    lm.c = 10.0

    lo = BinnedLineObs(m=true_m, c=true_c, N=10, binner_cls=SimpleBinner)

    opt = NestleOptimizer(num_live_points=50, observed=lo, model=lm)
    opt.enable_fit("m")
    opt.enable_fit("c")
    opt.set_boundary("m", [0.8 * true_m, 1.2 * true_m])
    opt.set_boundary("c", [0.8 * true_c, 1.2 * true_c])

    opt.fit()

    idx, optimized_map, optimized_median, values = next(opt.get_solution())
    opt.update_model(optimized_map)

    assert lm.m == pytest.approx(true_m, rel=0.2)
    assert lm.c == pytest.approx(true_c, rel=0.2)

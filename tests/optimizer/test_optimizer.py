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


def test_generate_profiles_limits_sample_workers_per_node(monkeypatch):
    """All ranks join reductions, but only capped ranks run model samples."""
    import taurex.mpi

    rank_state = {"rank": 0}
    rank_count = 8
    local_size = 4
    worker_limit = 2
    sample_list = [([sample], float(sample + 1)) for sample in range(10)]
    updated_samples = []
    compute_error_calls = []

    class LocalComm:
        def Get_rank(self):
            return rank_state["rank"] % local_size

        def Get_size(self):
            return local_size

    monkeypatch.setattr(taurex.mpi, "has_mpi", lambda: True)
    monkeypatch.setattr(taurex.mpi, "get_rank", lambda: rank_state["rank"])
    monkeypatch.setattr(taurex.mpi, "nprocs", lambda: rank_count)
    monkeypatch.setattr(taurex.mpi, "shared_comm", LocalComm)
    monkeypatch.setattr(taurex.mpi, "broadcast", lambda samples: sample_list)
    monkeypatch.setattr(
        taurex.mpi,
        "allgather",
        lambda is_worker: [
            mpi_rank % local_size < worker_limit for mpi_rank in range(rank_count)
        ],
    )

    class Model:
        def compute_error(self, samples, **kwargs):
            compute_error_calls.append(rank_state["rank"])
            return list(samples()), {}

    class FakeOptimizer:
        _uncertainty_max_workers_per_node = worker_limit
        _model = Model()
        _binner = object()

        def sample_parameters(self, solution):
            yield from sample_list

        def update_model(self, parameters):
            updated_samples.append(parameters[0])

        def info(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

    results = []
    for rank in range(rank_count):
        rank_state["rank"] = rank
        profile_result, spectrum_result = Optimizer.generate_profiles(
            FakeOptimizer(), solution=0, binning=[]
        )
        results.extend(profile_result)
        assert spectrum_result == {}

    assert sorted(updated_samples) == list(range(len(sample_list)))
    assert len(set(updated_samples)) == len(sample_list)
    assert compute_error_calls == list(range(rank_count))
    assert sorted(results) == [weight for _, weight in sample_list]


def test_generate_profiles_runs_without_mpi(monkeypatch):
    """Posterior profiles use one local worker when MPI is unavailable."""
    import taurex.mpi

    sample_list = [([sample], float(sample + 1)) for sample in range(3)]
    updated_samples = []

    monkeypatch.setattr(taurex.mpi, "has_mpi", lambda: False)
    monkeypatch.setattr(taurex.mpi, "get_rank", lambda: 0)
    monkeypatch.setattr(taurex.mpi, "nprocs", lambda: 1)
    monkeypatch.setattr(
        taurex.mpi, "shared_comm", lambda: pytest.fail("shared MPI comm requested")
    )
    monkeypatch.setattr(taurex.mpi, "broadcast", lambda samples: sample_list)
    monkeypatch.setattr(taurex.mpi, "allgather", lambda is_worker: [is_worker])

    class Model:
        def compute_error(self, samples, **kwargs):
            return list(samples()), {}

    class FakeOptimizer:
        _uncertainty_max_workers_per_node = 64
        _model = Model()
        _binner = object()

        def sample_parameters(self, solution):
            yield from sample_list

        def update_model(self, parameters):
            updated_samples.append(parameters[0])

        def info(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

    profile_result, spectrum_result = Optimizer.generate_profiles(
        FakeOptimizer(), solution=0, binning=[]
    )

    assert updated_samples == list(range(len(sample_list)))
    assert profile_result == [weight for _, weight in sample_list]
    assert spectrum_result == {}


@pytest.mark.parametrize("worker_limit", [0, -1, True, 1.5, "invalid"])
def test_optimizer_rejects_invalid_uncertainty_worker_limit(worker_limit):
    """The per-node worker limit must be a positive integer or None."""
    with pytest.raises(ValueError, match="uncertainty_max_workers_per_node"):
        Optimizer("test", uncertainty_max_workers_per_node=worker_limit)


def test_optimizer_accepts_parameter_file_worker_limit_values():
    """Convert ConfigObj string values to the configured worker limit."""
    optimizer = Optimizer("test")
    assert optimizer._uncertainty_max_workers_per_node == 64

    optimizer = Optimizer("test", uncertainty_max_workers_per_node=64.0)
    assert optimizer._uncertainty_max_workers_per_node == 64

    optimizer = Optimizer("test", uncertainty_max_workers_per_node="32")
    assert optimizer._uncertainty_max_workers_per_node == 32

    optimizer = Optimizer("test", uncertainty_max_workers_per_node="None")
    assert optimizer._uncertainty_max_workers_per_node is None


class _SolutionModel:
    def __init__(self):
        self.forward_evaluations = 0
        self.profile_generations = 0

    def model(self, **kwargs):
        self.forward_evaluations += 1
        return "spectrum"

    def generate_profiles(self):
        self.profile_generations += 1
        return {"temperature": {"median": 1000.0}}


class _SolutionBinner:
    def generate_spectrum_output(self, result, output_size):
        return {"spectrum": result}


class _FakeSolutionOptimizer:
    def __init__(self):
        self._model = _SolutionModel()
        self._binner = _SolutionBinner()
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

    fake = _FakeSolutionOptimizer()
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

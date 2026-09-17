"""Tests for ``DIFFERENTIABILITY.md`` and the JAX/PyTorch prototype.

Two groups of tests:

* **claims** -- assert that the structural obstacles the report lists really
  exist in the current source tree, and that a numba ``nopython`` kernel cannot
  consume a torch tensor;
* **prototype** -- assert that a faithful port of the forward model is
  differentiable in PyTorch and JAX, that the analytic gradients agree with
  central finite differences, and that the early-out shortcut really does break
  autodiff at its threshold.

The AD frameworks are optional: every test that needs one is skipped when it is
not installed, so the tauREx test matrix stays green.
"""

from __future__ import annotations

import functools
import importlib.util
import pathlib
import sys
import time

import numpy as np
import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PROTOTYPE_PATH = REPO_ROOT / "examples" / "differentiability" / "diffproto.py"


@functools.lru_cache(maxsize=1)
def load_prototype():
    """Load the prototype module from ``examples/`` by file path."""
    spec = importlib.util.spec_from_file_location("diffproto", PROTOTYPE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["diffproto"] = module
    spec.loader.exec_module(module)
    return module


proto = load_prototype()


# --------------------------------------------------------------------------
# group 1: does the report describe the current source correctly?
# --------------------------------------------------------------------------
def test_report_obstacles_exist_in_source():
    """Every structural obstacle listed in the report exists in the source."""
    src = REPO_ROOT / "src" / "taurex"
    transmission = (src / "model" / "transmission.py").read_text()
    optimizer = (src / "optimizer" / "optimizer.py").read_text()
    ktable = (src / "opacity" / "ktables" / "ktable.py").read_text()
    guillot = (src / "data" / "profiles" / "temperature" / "guillot.py").read_text()
    interpolate = (src / "opacity" / "interpolateopacity.py").read_text()
    util = (src / "util" / "util.py").read_text()

    # obstacle 5/7: numba kernel + einsum path for the optical depth
    assert "nopython=True" in (src / "contributions" / "contribution.py").read_text()
    assert (
        'np.einsum("ki,k,k->i"'
        in (src / "contributions" / "contribution.py").read_text()
    )

    # obstacle 6: data-dependent early-out inside the layer loop
    assert "if tau[layer].min() > 10:" in transmission

    # obstacle 8: in-place exp in compute_absorption
    assert "np.exp(-tau, out=tau)" in transmission

    # obstacle 9: errors swallowed into np.nan
    assert "except InvalidModelException:" in optimizer
    assert "return np.nan" in optimizer

    # obstacle 10: log-space transform of a fitting parameter
    assert "math.log10(self.value)" in optimizer

    # obstacle 2: searchsorted index lookup for the opacity grid
    assert "find_closest_pair" in interpolate
    assert "searchsorted" in util

    # obstacle 3: scipy interp1d rebinning of k-tables
    assert "from scipy.interpolate import interp1d" in ktable

    # obstacle 4: scipy special function in the temperature profile
    assert "spe.expn" in guillot


def test_numba_kernel_rejects_torch_tensor():
    """A numba ``nopython`` kernel cannot consume a torch tensor (obstacle 5)."""
    numba = pytest.importorskip("numba")
    torch = pytest.importorskip("torch")
    from taurex.contributions.contribution import contribute_tau

    assert isinstance(contribute_tau, numba.core.registry.CPUDispatcher)

    nlayers, ngrid, nk = 3, 4, 5
    tau = torch.zeros((nlayers, ngrid), dtype=torch.float64, requires_grad=True)
    with pytest.raises(numba.core.errors.NumbaError):
        contribute_tau(
            0,
            nk,
            0,
            torch.ones((nk, ngrid), dtype=torch.float64),
            np.ones(nk),
            np.ones((nlayers, nk)),
            nlayers,
            ngrid,
            0,
            tau,
        )


# --------------------------------------------------------------------------
# group 2: is the ported forward model actually differentiable?
# --------------------------------------------------------------------------
def test_numpy_and_torch_forward_agree():
    """The torch port reproduces the numpy reference spectrum."""
    torch = pytest.importorskip("torch")
    theta = torch.tensor(proto.THETA0, dtype=torch.float64, requires_grad=True)
    model = proto.forward_torch(theta).detach().cpu().numpy()
    np.testing.assert_allclose(model, proto.forward_np(proto.THETA0), rtol=1e-12)


def test_torch_loop_matches_vectorised():
    """The eager per-layer ``+=`` loop gives the same values and gradients."""
    torch = pytest.importorskip("torch")

    def grad_of(loop: bool) -> np.ndarray:
        theta = torch.tensor(proto.THETA0, dtype=torch.float64, requires_grad=True)
        proto.chisq_torch(theta, loop=loop).backward()
        return theta.grad.cpu().numpy()

    np.testing.assert_allclose(grad_of(True), grad_of(False), rtol=1e-9)


def test_torch_grad_matches_finite_difference():
    """Autograd through the whole chi-squared matches central differences."""
    torch = pytest.importorskip("torch")
    theta = torch.tensor(proto.THETA0, dtype=torch.float64, requires_grad=True)
    proto.chisq_torch(theta).backward()
    finite_difference = proto.finite_difference_grad(proto.chisq_np, proto.THETA0)
    np.testing.assert_allclose(
        theta.grad.cpu().numpy(), finite_difference, rtol=1e-5, atol=1e-8
    )


def test_torch_jacobian_matches_finite_difference():
    """The whole spectrum Jacobian is differentiable w.r.t. all parameters."""
    torch = pytest.importorskip("torch")
    theta = torch.tensor(proto.THETA0, dtype=torch.float64, requires_grad=True)
    jac = (
        torch.autograd.functional.jacobian(
            lambda t: proto.forward_torch(t), theta, vectorize=False
        )
        .detach()
        .cpu()
        .numpy()
        .T  # torch gives (n_outputs, n_params)
    )
    finite_difference = np.empty_like(jac)
    for i in range(proto.THETA0.size):
        step = 1e-6 * max(1.0, abs(proto.THETA0[i]))
        up = proto.THETA0.copy()
        down = proto.THETA0.copy()
        up[i] += step
        down[i] -= step
        finite_difference[i] = (proto.forward_np(up) - proto.forward_np(down)) / (
            2.0 * step
        )
    np.testing.assert_allclose(jac, finite_difference, rtol=1e-5, atol=1e-12)


def test_jax_forward_matches_numpy():
    """The JAX port reproduces the numpy reference spectrum."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    proto._jax()
    model = np.asarray(
        jax.jit(lambda t: proto.forward_jax(t))(jnp.asarray(proto.THETA0))
    )
    # 1e-11 rather than 1e-12: the JAX `E_2` is a bounded series/continued-fraction
    # hybrid with 2.5e-13 worst relative error, which sets the comparison floor.
    np.testing.assert_allclose(model, proto.forward_np(proto.THETA0), rtol=1e-11)


def test_jax_grad_matches_finite_difference():
    """``jax.grad`` of the chi-squared matches central differences."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    proto._jax()
    grad = jax.jit(jax.grad(lambda t: proto.chisq_jax(t)))(jnp.asarray(proto.THETA0))
    finite_difference = proto.finite_difference_grad(proto.chisq_np, proto.THETA0)
    np.testing.assert_allclose(
        np.asarray(grad), finite_difference, rtol=1e-5, atol=1e-8
    )


def test_jax_matches_torch_gradient():
    """Both frameworks agree with each other far below the FD accuracy."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    proto._jax()
    theta = torch.tensor(proto.THETA0, dtype=torch.float64, requires_grad=True)
    proto.chisq_torch(theta).backward()
    jax_grad = jax.grad(lambda t: proto.chisq_jax(t))(jnp.asarray(proto.THETA0))
    np.testing.assert_allclose(
        theta.grad.cpu().numpy(), np.asarray(jax_grad), rtol=1e-9, atol=1e-12
    )


# --------------------------------------------------------------------------
# group 3: the two genuinely hard problems from the report
# --------------------------------------------------------------------------
def test_torch_expn_matches_scipy():
    """The custom ``E_2`` autograd rule reproduces scipy in the forward pass."""
    torch = pytest.importorskip("torch")
    x = torch.linspace(0.1, 5.0, 8, dtype=torch.float64, requires_grad=True)
    np.testing.assert_allclose(
        proto.expn2_torch(x).detach().cpu().numpy(),
        proto.scipy_expn(2, x.detach().cpu().numpy()),
        rtol=1e-13,
    )


def test_torch_expn_gradcheck():
    """Gradcheck validates the ``dE_2/dx = -E_1(x)`` backward rule."""
    torch = pytest.importorskip("torch")
    x = torch.linspace(0.1, 5.0, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(proto.expn2_torch, (x,), eps=1e-6, atol=1e-8)


def test_jax_expn_is_differentiable():
    """The JAX ``expn`` derivative matches the torch custom rule."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    proto._jax()
    x_np = np.linspace(0.1, 5.0, 8)
    jax_val, jax_grad = jax.jit(
        jax.value_and_grad(lambda v: jnp.sum(proto.expn2_jax(v)))
    )(jnp.asarray(x_np))
    x_t = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    torch_val = torch.sum(proto.expn2_torch(x_t))
    torch_grad = torch.autograd.grad(torch_val, x_t)[0]
    np.testing.assert_allclose(float(jax_val), float(torch_val.detach()), rtol=1e-11)
    np.testing.assert_allclose(
        np.asarray(jax_grad), torch_grad.cpu().numpy(), rtol=1e-10
    )


def test_jax_expn2_matches_scipy_over_real_range():
    """The bounded ``E_2`` agrees with scipy across the Guillot argument range."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    proto._jax()
    gamma_tau = np.logspace(-9.0, 4.5, 200)
    values = np.asarray(jax.jit(proto.expn2_jax)(jnp.asarray(gamma_tau)))
    np.testing.assert_allclose(
        values, proto.scipy_expn(2, gamma_tau), rtol=1e-12, atol=1e-19
    )


def test_jax_expn2_avoids_the_slow_path():
    """``E_2`` stays fast on the mixed range that makes JAX's ``expn`` crawl."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    proto._jax()
    # jax.scipy.special.expn(2, [1e-8, 1.0]) needs ~74 s in the same setup.
    mixed = jnp.asarray(np.array([1e-8, 1.0, 5.0e4]))
    start = time.perf_counter()
    jax.jit(proto.expn2_jax)(mixed).block_until_ready()
    assert time.perf_counter() - start < 5.0


def test_early_out_gate_breaks_autodiff():
    """The ``if tau > 10: continue`` shortcut makes FD and AD disagree."""
    torch = pytest.importorskip("torch")
    x = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    ad = torch.autograd.grad(proto.saturation_gate_torch(x), x)[0].item()
    step = 1e-3
    fd = (
        proto.saturation_gate_np(1.0 + step) - proto.saturation_gate_np(1.0 - step)
    ) / (2.0 * step)
    # The function is discontinuous at x = 10/9.999: AD silently returns the
    # derivative of the branch it happens to be on, FD straddles the kink.
    assert ad > 10.0
    assert fd < -100.0


def test_detached_gate_keeps_local_derivative():
    """Away from the threshold the gate is differentiable in both frameworks."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    proto._jax()
    x = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    ad = torch.autograd.grad(proto.saturation_gate_torch(x), x)[0].item()
    jax_grad = float(
        jax.grad(lambda v: proto.saturation_gate_jax(v, 1.5))(jnp.asarray(1.5))
    )
    step = 1e-6
    fd = (
        proto.saturation_gate_np(1.5 + step) - proto.saturation_gate_np(1.5 - step)
    ) / (2.0 * step)
    np.testing.assert_allclose([ad, jax_grad], [fd, fd], rtol=1e-5)


def test_jax_rejects_python_if_on_traced_value():
    """A Python ``if`` on a traced value fails loudly under ``jit``.

    Note that plain ``jax.grad`` does *not* raise here in JAX 0.11 -- it takes
    one branch and silently returns its derivative, which is exactly the trap
    the report warns about. ``jax.jit(jax.grad(...))`` is the realistic usage.
    """

    def traced_gate(x):
        tau = 9.999 * x
        if tau > 10.0:
            return tau
        return tau + 5.0 * x

    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    proto._jax()
    with pytest.raises(Exception) as excinfo:
        jax.jit(jax.grad(traced_gate))(jnp.asarray(1.0))
    message = str(excinfo.value)
    assert "Tracer" in message or "concret" in message


# --------------------------------------------------------------------------
# group 4: benchmark
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_benchmark_returns_timings():
    """The benchmark runs for every installed framework and returns timings."""
    pytest.importorskip("torch")
    pytest.importorskip("jax")
    results = proto.benchmark(repeats=2)
    labels = [label for label, _ in results]
    assert all(millis > 0.0 for _, millis in results)
    assert any("torch" in label for label in labels)
    assert any("jax" in label for label in labels)

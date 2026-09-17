"""Minimal differentiable TauREx-style transmission forward model.

This is a self-contained prototype used to test the claims of
``DIFFERENTIABILITY.md``: the TauREx3 forward mathematics is differentiable, and
the barriers to reverse-mode autodiff are *structural*, not mathematical.

The physics is deliberately simplified (synthetic single-gas opacity, fixed
altitude grid) but the shape of the computation, and the location of every
non-differentiable construct, matches the real code:

===========================================================  ===============================
report obstacle                                              reproduced here by
===========================================================  ===============================
2. ``searchsorted`` opacity grid index lookup                :func:`closest_pair_np`,
                                                             :func:`interp_xsec_torch`
4. ``scipy.special.expn`` in ``Guillot2010.profile``         :func:`guillot_temperature_np`
                                                             / ``_expn2`` (torch AD rule),
                                                             :func:`expn2_jax` (bounded)
5. numba ``nopython`` optical-depth kernels                  tested directly in
                                                             ``tests/differentiability``
6. ``if tau[layer].min() > 10: continue``                    :func:`saturation_gate_np`
7. ``for layer ... tau[layer] += ...`` in-place loop         ``loop=True`` in the forwards
8. ``Guillot2010._check_values`` raising on invalid input    clamped ``t4 ** 0.25``
9. ``np.exp(-tau, out=tau)``                                 ``trans = exp(-tau)``
10. ``math.log10(value)`` on fitting parameters               ``theta`` is a tensor
===========================================================  ===============================

One obstacle the report does *not* list: ``jax.scipy.special.expn`` is unusable for the
Guillot profile. Its branches all evaluate over the whole array, and a batch that mixes
arguments below and above 1 sends an internal ``while_loop`` into a crawl -- measured at
**74 s for a two-element array** ``[1e-8, 1.0]``. ``gamma * tau`` spans ~1e-9 to ~1e4
across the layers, so the real profile hits that mix immediately. :func:`expn2_jax` instead
uses a 40-term convergent series below ``x = 3`` and a 48-term Gauss continued fraction
above it, both with a fixed iteration count: measured 2.5e-13 worst relative error and
~0.3 ms for the profile's argument range, against ~7.2 ms for the clipped ``exp1`` route
(a 23x difference that has nothing to do with autodiff and everything to do with JAX's
standard library).

Run ``python examples/differentiability/diffproto.py`` for a gradient check plus
a small CPU/GPU benchmark.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.special import expn as scipy_expn


# --------------------------------------------------------------------------
# Configuration. Values mirror the real TransmissionModel defaults where it
# matters (pressure bounds, 40 layers, transit geometry).
# --------------------------------------------------------------------------
NLAYERS = 40
NWN = 300

P_MIN = 1e-4
P_MAX = 1e6

T_GRID = np.linspace(200.0, 8000.0, 40)
P_GRID = np.logspace(np.log10(P_MIN), np.log10(P_MAX), 30)
LOG_P_GRID = np.log10(P_GRID)

R_PLANET = 8.0e7
R_STAR = 7.0e8
GRAVITY = 20.0
T_INT = 100.0
KAPPA_V1 = 1e-3
KAPPA_V2 = 1.0
T_REF = 1200.0
MEAN_MOLAR_MASS = 2.3e-3
K_B = 1.380649e-23
N_A = 6.02214076e23
XS_SCALE = 1e-30

WNG_GRID = np.logspace(-6.0, -5.0, NWN)

#: The cross-section grid holds :data:`NGAS_MAX` gases and a model with ``ngas``
#: gases simply slices the first ``ngas`` of them. One grid therefore serves
#: every model size the scaling experiment needs.
NGAS_MAX = 8
GAS_BAND_STEP = 0.7
NGAS = 1

ERR = 1e-3

_TORCH_CACHE: dict = {}
_JAX_STATE: dict = {}


# --------------------------------------------------------------------------
# numpy reference implementation
# --------------------------------------------------------------------------
def pressure_profile(
    nlayers: int = NLAYERS, pmin: float = P_MIN, pmax: float = P_MAX
) -> tuple[np.ndarray, np.ndarray]:
    """Return layer-boundary and layer-centre pressures [Pa]."""
    p_bound = np.logspace(np.log10(pmin), np.log10(pmax), nlayers + 1)
    p_center = np.sqrt(p_bound[:-1] * p_bound[1:])
    return p_bound, p_center


def altitude_profile(
    p_bound: np.ndarray, nlayers: int = NLAYERS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boundary/centre altitudes and layer thicknesses [m]."""
    scale_height = K_B * T_REF / (MEAN_MOLAR_MASS / N_A * GRAVITY)
    z_bound = scale_height * np.log(p_bound / p_bound[0])
    z_center = 0.5 * (z_bound[:-1] + z_bound[1:])
    return z_bound, z_center, np.diff(z_bound)


def build_path_matrix(z_bound: np.ndarray, z_center: np.ndarray) -> np.ndarray:
    """Return the ``(nlayers, nlayers)`` chord-length matrix [m].

    Entry ``[layer, k]`` is the length of the ray with impact parameter
    ``R_planet + z_center[layer]`` inside shell ``k``. Same construction as
    ``TransmissionModel.compute_path_length_old`` but fully vectorised.
    """
    b = (R_PLANET + z_center)[:, None]
    r0 = R_PLANET + z_bound[None, :-1]
    r1 = R_PLANET + z_bound[None, 1:]
    m0 = np.clip(r0**2 - b**2, 0.0, None)
    m1 = np.clip(r1**2 - b**2, 0.0, None)
    return 2.0 * (np.sqrt(m1) - np.sqrt(m0))


def build_xsec_grid(ngas_max: int = NGAS_MAX) -> np.ndarray:
    """Return synthetic ``(gas, T, P, wavenumber)`` cross sections.

    Units are m^2 per molecule. Gas ``i`` has its band pattern shifted, so every
    gas parameter leaves a distinct spectral signature.

    The band structure is deliberately *not* separable in ``(T, wavenumber)``:
    a band pattern that shifts with temperature is what makes the spectral shape
    depend on the atmospheric parameters. A separable power law would leave the
    spectrum shape almost parameter-independent, the Jacobian rank deficient and
    both the MAP fit and the Fisher matrix meaningless.
    """
    gas = np.arange(ngas_max)[:, None, None, None]
    t = (T_GRID / 1000.0)[:, None, None]
    p = (P_GRID / 1e5)[None, :, None]
    w = (WNG_GRID / WNG_GRID.mean())[None, None, :]
    bands = 1.0 + 0.6 * np.cos(8.0 * np.log(w) + 2.0 * np.log(t) + GAS_BAND_STEP * gas)
    return XS_SCALE * t**-1.5 * p**0.25 * w**-1.2 * bands


P_BOUND, P_CENTER = pressure_profile()
Z_BOUND, Z_CENTER, DZ = altitude_profile(P_BOUND)
PATH = build_path_matrix(Z_BOUND, Z_CENTER)
XSEC_GRID = build_xsec_grid()
LOG_P_CENTER = np.log10(P_CENTER)


def closest_pair_np(arr: np.ndarray, value: float) -> tuple[int, int]:
    """Port of ``taurex.util.find_closest_pair`` (searchsorted index lookup)."""
    right = int(np.searchsorted(arr, value))
    right = max(min(arr.shape[0] - 1, right), 1)
    return max(0, right - 1), right


def guillot_temperature_np(
    pressure: np.ndarray, t_irr: float, alpha: float, kappa_ir: float
) -> np.ndarray:
    """Guillot (2010) profile, including the ``scipy.special.expn`` term."""
    tau = kappa_ir * pressure / GRAVITY
    gamma_1 = KAPPA_V1 / kappa_ir
    gamma_2 = KAPPA_V2 / kappa_ir

    def eta(gamma: float) -> np.ndarray:
        part1 = 2.0 / 3.0 + 2.0 / (3.0 * gamma) * (
            1.0 + (gamma * tau / 2.0 - 1.0) * np.exp(-gamma * tau)
        )
        part2 = 2.0 * gamma / 3.0 * (1.0 - tau**2 / 2.0) * scipy_expn(2, gamma * tau)
        return part1 + part2

    t4 = (
        0.75 * T_INT**4 * (2.0 / 3.0 + tau)
        + 0.75 * t_irr**4 * (1.0 - alpha) * eta(gamma_1)
        + 0.75 * t_irr**4 * alpha * eta(gamma_2)
    )
    # Guillot2010._check_values raises InvalidModelException when T < 0.
    # Reverse-mode AD needs a branch the graph survives, so clamp instead.
    return np.clip(t4, 0.0, None) ** 0.25


def interp_xsec_np(
    temperature: np.ndarray, log_pressure: np.ndarray, xsec: np.ndarray, ngas: int
) -> np.ndarray:
    """Bilinear ``(T, log10 P)`` cross-section interpolation for ``ngas`` gases."""
    out = np.empty((ngas, temperature.size, xsec.shape[-1]))
    for layer in range(temperature.size):
        t_min, t_max = closest_pair_np(T_GRID, temperature[layer])
        p_min, p_max = closest_pair_np(LOG_P_GRID, log_pressure[layer])
        fx = (temperature[layer] - T_GRID[t_min]) / (T_GRID[t_max] - T_GRID[t_min])
        fy = (log_pressure[layer] - LOG_P_GRID[p_min]) / (
            LOG_P_GRID[p_max] - LOG_P_GRID[p_min]
        )
        f00 = xsec[:ngas, t_min, p_min]
        f10 = xsec[:ngas, t_max, p_min]
        f01 = xsec[:ngas, t_min, p_max]
        f11 = xsec[:ngas, t_max, p_max]
        out[:, layer] = (1.0 - fy) * ((1.0 - fx) * f00 + fx * f10) + fy * (
            (1.0 - fx) * f01 + fx * f11
        )
    return out


def absorption_from_tau_np(
    tau: np.ndarray, z_center: np.ndarray, dz: np.ndarray
) -> np.ndarray:
    """Port of ``TransmissionModel.compute_absorption`` (no in-place exp)."""
    trans = np.exp(-tau)
    integral = np.sum(
        (R_PLANET + z_center)[:, None] * (1.0 - trans) * dz[:, None] * 2.0, axis=0
    )
    return (R_PLANET**2 + integral) / R_STAR**2


def forward_np(theta: np.ndarray, loop: bool = False) -> np.ndarray:
    """Reference transmission spectrum for ``ngas + 3`` fitting parameters.

    ``theta = [log10_X_1 .. log10_X_ngas, T_irr, alpha, kappa_ir]``.
    """
    ngas = int(np.size(theta)) - 3
    mixing_ratios = 10.0 ** theta[:ngas]
    t_irr, alpha, kappa_ir = theta[ngas], theta[ngas + 1], theta[ngas + 2]
    temperature = guillot_temperature_np(P_CENTER, t_irr, alpha, kappa_ir)
    sigma_gas = interp_xsec_np(temperature, LOG_P_CENTER, XSEC_GRID, ngas)
    sigma_mix = np.einsum("i,ilw->lw", mixing_ratios, sigma_gas)
    density = P_CENTER / (K_B * temperature)
    if loop:
        tau = np.zeros((NLAYERS, WNG_GRID.size))
        for layer in range(NLAYERS):
            for k in range(NLAYERS):
                tau[layer] += PATH[layer, k] * density[k] * sigma_mix[k]
    else:
        tau = np.einsum("lk,k,kw->lw", PATH, density, sigma_mix)
    return absorption_from_tau_np(tau, Z_CENTER, DZ)


def theta_names(theta: np.ndarray) -> tuple[str, ...]:
    """Return the fitting-parameter names for a ``theta`` vector."""
    ngas = int(np.size(theta)) - 3
    return tuple(
        [f"log10_X{i + 1}" for i in range(ngas)] + ["T_irr", "alpha", "kappa_ir"]
    )


def build_dataset(ngas: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(theta0, theta_data, spectrum)`` for an ``ngas``-gas model.

    ``ngas=1`` reproduces the four-parameter model used everywhere else.
    """
    offsets = -0.1 * np.arange(ngas)
    theta0 = np.concatenate([np.full(ngas, -3.0) + offsets, [1200.0, 0.5, 1.0e-3]])
    theta_data = theta0 + np.concatenate([np.full(ngas, 0.05), [30.0, 0.02, 1.0e-4]])
    return theta0, theta_data, forward_np(theta_data)


THETA0, THETA_DATA, DATA = build_dataset(NGAS)
THETA_NAMES = theta_names(THETA0)


def chisq_np(
    theta: np.ndarray, loop: bool = False, data: np.ndarray | None = None
) -> float:
    """Chi-squared of the reference model against the synthetic data."""
    target = DATA if data is None else data
    residual = (target - forward_np(theta, loop=loop)) / ERR
    return float(np.sum(residual * residual))


def finite_difference_grad(func, theta: np.ndarray, rel: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of a scalar callable of ``theta``."""
    grad = np.zeros_like(theta)
    for i in range(theta.size):
        step = rel * max(1.0, abs(theta[i]))
        up = theta.copy()
        down = theta.copy()
        up[i] += step
        down[i] -= step
        grad[i] = (func(up) - func(down)) / (2.0 * step)
    return grad


def saturation_gate_np(
    x: float, decision_x: float | None = None, limit: float = 10.0
) -> float:
    """Reproduce the ``if tau[layer].min() > 10: continue`` shortcut.

    Two contributions are added to one layer; the second is dropped when the
    accumulated optical depth is already saturated. This is exactly the
    coupling that ``TransmissionModel.path_integral`` introduces.

    Args:
        x: scaling applied to both contributions (the differentiated input).
        decision_x: value used for the branch decision. ``None`` uses ``x``
            itself, i.e. the coupled (discontinuous) behaviour.
        limit: saturation threshold.

    Returns:
        The gated optical depth.
    """
    decide = x if decision_x is None else decision_x
    saturated = 9.999 * decide > limit
    tau = 9.999 * x
    if saturated:
        return float(tau)
    return float(tau + 5.0 * x)


# --------------------------------------------------------------------------
# PyTorch implementation
# --------------------------------------------------------------------------
def _torch():
    """Import and return :mod:`torch` (raises if it is not installed)."""
    import torch

    return torch


def _torch_arrays(device=None) -> dict:
    """Build (and cache) the float64 torch copies of every grid."""
    torch = _torch()
    key = ("arrays", None if device is None else str(device))
    if key not in _TORCH_CACHE:
        dtype = torch.float64
        arrays = {
            "path": torch.as_tensor(PATH, dtype=dtype),
            "xsec": torch.as_tensor(XSEC_GRID, dtype=dtype),
            "pressure": torch.as_tensor(P_CENTER, dtype=dtype),
            "z_center": torch.as_tensor(Z_CENTER, dtype=dtype),
            "dz": torch.as_tensor(DZ, dtype=dtype),
            "t_grid": torch.as_tensor(T_GRID, dtype=dtype),
            "log_p_grid": torch.as_tensor(LOG_P_GRID, dtype=dtype),
            "data": torch.as_tensor(DATA, dtype=dtype),
        }
        if device is not None:
            arrays = {k: v.to(device) for k, v in arrays.items()}
        _TORCH_CACHE[key] = arrays
    return _TORCH_CACHE[key]


def _expn2_class():
    """Build (once) the ``torch.autograd.Function`` for ``E_2``.

    PyTorch has no ``expn``; the forward value comes from scipy and the
    backward pass uses the exact recurrence ``dE_n/dx = -E_{n-1}(x)``.
    """
    if "expn2" not in _TORCH_CACHE:
        torch = _torch()

        class _Expn2(torch.autograd.Function):
            """``E_2(x)`` with the analytic ``-E_1(x)`` derivative.

            Written in the modern ``setup_context`` style so that it survives
            ``torch.func`` transforms such as ``jacrev``. Forward mode
            (``jacfwd``, ``vmap``) additionally needs a vmappable *forward*,
            which a host-side scipy call cannot provide: reaching for scipy
            inside the primal costs you torch.func composability.
            """

            @staticmethod
            def forward(x):
                """Evaluate ``E_2`` with scipy (no graph is built)."""
                values = scipy_expn(2, x.detach().cpu().numpy())
                return torch.as_tensor(values, dtype=x.dtype, device=x.device)

            @staticmethod
            def setup_context(ctx, inputs, output):
                """Store the input for the backward pass."""
                (x,) = inputs
                ctx.save_for_backward(x)

            @staticmethod
            def backward(ctx, grad_output):
                """Backpropagate with ``dE_2/dx = -E_1(x)``."""
                (x,) = ctx.saved_tensors
                deriv = -scipy_expn(1, x.detach().cpu().numpy())
                return grad_output * torch.as_tensor(
                    deriv, dtype=x.dtype, device=x.device
                )

        _TORCH_CACHE["expn2"] = _Expn2
    return _TORCH_CACHE["expn2"]


def expn2_torch(x):
    """``E_2(x)`` for torch tensors, differentiable through a custom rule."""
    return _expn2_class().apply(x)


def guillot_temperature_torch(pressure, t_irr, alpha, kappa_ir):
    """Torch port of :func:`guillot_temperature_np`."""
    import torch

    tau = kappa_ir * pressure / GRAVITY
    gamma_1 = KAPPA_V1 / kappa_ir
    gamma_2 = KAPPA_V2 / kappa_ir

    def eta(gamma):
        part1 = 2.0 / 3.0 + 2.0 / (3.0 * gamma) * (
            1.0 + (gamma * tau / 2.0 - 1.0) * torch.exp(-gamma * tau)
        )
        part2 = 2.0 * gamma / 3.0 * (1.0 - tau**2 / 2.0) * expn2_torch(gamma * tau)
        return part1 + part2

    t4 = (
        0.75 * T_INT**4 * (2.0 / 3.0 + tau)
        + 0.75 * t_irr**4 * (1.0 - alpha) * eta(gamma_1)
        + 0.75 * t_irr**4 * alpha * eta(gamma_2)
    )
    return torch.clamp(t4, min=0.0) ** 0.25


def interp_xsec_torch(temperature, log_pressure, arrays, ngas) -> object:
    """Vectorised torch port of :func:`interp_xsec_np` (all layers at once).

    The ``searchsorted`` indices are integer valued, so they carry no gradient;
    they are detached on purpose and the gradient flows through the linear
    interpolation weights. This is the standard fix for report obstacle 2.
    """
    import torch

    xsec = arrays["xsec"]
    t_grid = arrays["t_grid"]
    p_grid = arrays["log_p_grid"]

    t_idx = torch.clamp(
        torch.searchsorted(t_grid, temperature.detach()), 1, t_grid.shape[0] - 1
    ).detach()
    p_idx = torch.clamp(
        torch.searchsorted(p_grid, log_pressure.detach()), 1, p_grid.shape[0] - 1
    ).detach()
    t_min, p_min = t_idx - 1, p_idx - 1

    fx = (temperature - t_grid[t_min]) / (t_grid[t_idx] - t_grid[t_min])
    fy = (log_pressure - p_grid[p_min]) / (p_grid[p_idx] - p_grid[p_min])
    f00 = xsec[:ngas, t_min, p_min]
    f10 = xsec[:ngas, t_idx, p_min]
    f01 = xsec[:ngas, t_min, p_idx]
    f11 = xsec[:ngas, t_idx, p_idx]
    return (1.0 - fy)[:, None] * ((1.0 - fx)[:, None] * f00 + fx[:, None] * f10) + fy[
        :, None
    ] * ((1.0 - fx)[:, None] * f01 + fx[:, None] * f11)


def _tau_loop_torch(density, sigma, path, use_early_out: bool = False):
    """Port of the ``for layer ... tau[layer] += ...`` accumulation."""
    import torch

    nlayers = density.shape[0]
    tau = torch.zeros((nlayers, sigma.shape[1]), dtype=sigma.dtype, device=sigma.device)
    for layer in range(nlayers):
        if use_early_out and bool(tau[layer].min().detach() > 10.0):
            continue
        for k in range(nlayers):
            tau[layer] = tau[layer] + path[layer, k] * density[k] * sigma[k]
    return tau


def forward_torch(theta, loop: bool = False, use_early_out: bool = False, arrays=None):
    """Torch port of :func:`forward_np`; ``theta`` must require grad."""
    import torch

    if arrays is None:
        arrays = _torch_arrays(theta.device)
    ngas = int(theta.shape[0]) - 3
    pressure = arrays["pressure"]
    temperature = guillot_temperature_torch(
        pressure, theta[ngas], theta[ngas + 1], theta[ngas + 2]
    )
    sigma_gas = interp_xsec_torch(temperature, torch.log10(pressure), arrays, ngas)
    sigma_mix = torch.einsum("i,ilw->lw", 10.0 ** theta[:ngas], sigma_gas)
    density = pressure / (K_B * temperature)
    if loop:
        tau = _tau_loop_torch(density, sigma_mix, arrays["path"], use_early_out)
    else:
        tau = torch.einsum("lk,k,kw->lw", arrays["path"], density, sigma_mix)
    trans = torch.exp(-tau)
    integral = torch.sum(
        (R_PLANET + arrays["z_center"])[:, None]
        * (1.0 - trans)
        * arrays["dz"][:, None]
        * 2.0,
        dim=0,
    )
    return (R_PLANET**2 + integral) / R_STAR**2


def chisq_torch(theta, loop: bool = False, arrays=None, data=None):
    """Chi-squared of the torch model against the synthetic data."""
    if arrays is None:
        arrays = _torch_arrays(theta.device)
    target = arrays["data"] if data is None else data
    residual = (target - forward_torch(theta, loop=loop, arrays=arrays)) / ERR
    return (residual * residual).sum()


def saturation_gate_torch(x, decision_x=None, limit: float = 10.0):
    """Torch port of :func:`saturation_gate_np` (eager, branch on detach)."""
    decide = x.detach() if decision_x is None else decision_x
    tau = 9.999 * x
    if bool(9.999 * decide > limit):
        return tau
    return tau + 5.0 * x


# --------------------------------------------------------------------------
# JAX implementation
# --------------------------------------------------------------------------
def _jax():
    """Import :mod:`jax` and enable float64 once."""
    import os

    # The prototype keeps torch and JAX resident in the same process. On a
    # consumer GPU XLA's default 75% preallocation makes the two fight for
    # memory, so grow the pool on demand instead.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax

    if not _JAX_STATE.get("x64"):
        jax.config.update("jax_enable_x64", True)
        _JAX_STATE["x64"] = True
    return jax


def _jax_arrays() -> dict:
    """Return the float64 JAX copies of every grid.

    The arrays are rebuilt on every call on purpose: caching ``jnp`` arrays that
    were created inside a ``jit`` trace leaks tracers into global state and
    raises ``jax.errors.UnexpectedTracerError`` on the next trace. For grids of
    this size the conversion cost is negligible.
    """
    import jax.numpy as jnp

    _jax()
    return {
        "path": jnp.asarray(PATH),
        "xsec": jnp.asarray(XSEC_GRID),
        "pressure": jnp.asarray(P_CENTER),
        "z_center": jnp.asarray(Z_CENTER),
        "dz": jnp.asarray(DZ),
        "t_grid": jnp.asarray(T_GRID),
        "log_p_grid": jnp.asarray(LOG_P_GRID),
        "data": jnp.asarray(DATA),
    }


EULER_GAMMA = 0.5772156649015329
SERIES_TERMS = 40
#: ``E_2`` uses the convergent series below this argument and the continued
#: fraction above it. Measured worst relative error of the hybrid: 2.5e-13 over
#: the whole argument range the Guillot profile produces (1e-9 to 1e5).
E2_SERIES_MAX = 3.0
E2_CF_TERMS = 48


def _exp1_series(x):
    """``E_1(x)`` from the convergent series (A&S 5.1.11).

    Accurate for arguments up to a few; cancellation takes over beyond that.
    """
    import jax.numpy as jnp
    from jax.scipy import special as jspecial

    k = jnp.arange(1.0, SERIES_TERMS + 1.0)
    log_factorial = jspecial.gammaln(k + 1.0)
    signs = jnp.where(k % 2.0 == 1.0, 1.0, -1.0)
    log_x = jnp.log(x)
    terms = jnp.exp(jnp.expand_dims(log_x, -1) * k - log_factorial) * signs / k
    total = -EULER_GAMMA - log_x + jnp.sum(terms, axis=-1)
    return jnp.reshape(total, jnp.shape(x))


def _expn2_continued_fraction(x):
    """``E_2(x)`` from the Gauss continued fraction by backward recursion.

    Evaluated from the last term backwards, with a *fixed* term count: a data
    dependent ``while_loop`` is exactly what makes JAX's own ``expn`` crawl, and
    the fixed count keeps the traced graph identical for every batch.

    Deliberately *not* Lentz's forward algorithm. Lentz needs a tiny seed
    (``1e-300``) which blows up to ``~1e300`` in the first step; the primal is
    fine because it cancels, but forward-mode autodiff differentiates that
    intermediate and the tangent of ``a / c`` with ``c ~ 1e-300`` divides by
    ``c**2 ~ 0`` and yields NaN. Backward recursion keeps every intermediate of
    order ``x + k``, so it differentiates cleanly.
    """
    import jax.numpy as jnp
    from jax import lax

    def body(i, term):
        k = E2_CF_TERMS - i
        b = x + 2.0 * k
        a_next = -(k + 1.0) * k
        return jnp.where(i == 0, b, b + a_next / term)

    term = lax.fori_loop(0, E2_CF_TERMS, body, jnp.zeros_like(x))
    return jnp.exp(-x) / term


def expn2_jax(x):
    """``E_2(x)`` without JAX's unbounded loops.

    ``jax.scipy.special.expn`` cannot be used here. Measuring it: a batch that
    mixes arguments below and above 1 sends an internal ``while_loop`` into a
    crawl (74 s for the two-element array ``[1e-8, 1.0]``), and the Guillot
    profile produces exactly that mix because ``gamma * tau`` spans ~1e-9 to
    ~1e4 across the layers.

    This implementation is bounded (a 40-term series below :data:`E2_SERIES_MAX`,
    a 48-term continued fraction above), autodiff-compatible, and measured at
    ~0.3 ms for the profile's argument range against ~7.2 ms for the clipped
    ``exp1`` route it replaces.
    """
    import jax.numpy as jnp

    x = jnp.maximum(x, 1.0e-300)
    series = jnp.exp(-x) - x * _exp1_series(x)
    return jnp.where(x < E2_SERIES_MAX, series, _expn2_continued_fraction(x))


def guillot_temperature_jax(pressure, t_irr, alpha, kappa_ir):
    """JAX port of :func:`guillot_temperature_np`."""
    import jax.numpy as jnp

    tau = kappa_ir * pressure / GRAVITY
    gamma_1 = KAPPA_V1 / kappa_ir
    gamma_2 = KAPPA_V2 / kappa_ir

    def eta(gamma):
        part1 = 2.0 / 3.0 + 2.0 / (3.0 * gamma) * (
            1.0 + (gamma * tau / 2.0 - 1.0) * jnp.exp(-gamma * tau)
        )
        part2 = 2.0 * gamma / 3.0 * (1.0 - tau**2 / 2.0) * expn2_jax(gamma * tau)
        return part1 + part2

    t4 = (
        0.75 * T_INT**4 * (2.0 / 3.0 + tau)
        + 0.75 * t_irr**4 * (1.0 - alpha) * eta(gamma_1)
        + 0.75 * t_irr**4 * alpha * eta(gamma_2)
    )
    return jnp.clip(t4, 0.0, None) ** 0.25


def interp_xsec_jax(temperature, log_pressure, arrays, ngas):
    """JAX port of :func:`interp_xsec_torch` with stopped-gradient indices."""
    import jax.numpy as jnp
    from jax import lax

    xsec = arrays["xsec"]
    t_grid = arrays["t_grid"]
    p_grid = arrays["log_p_grid"]
    t_idx = lax.stop_gradient(
        jnp.clip(jnp.searchsorted(t_grid, temperature), 1, t_grid.shape[0] - 1)
    )
    p_idx = lax.stop_gradient(
        jnp.clip(jnp.searchsorted(p_grid, log_pressure), 1, p_grid.shape[0] - 1)
    )
    t_min, p_min = t_idx - 1, p_idx - 1

    fx = (temperature - t_grid[t_min]) / (t_grid[t_idx] - t_grid[t_min])
    fy = (log_pressure - p_grid[p_min]) / (p_grid[p_idx] - p_grid[p_min])
    f00 = xsec[:ngas, t_min, p_min]
    f10 = xsec[:ngas, t_idx, p_min]
    f01 = xsec[:ngas, t_min, p_idx]
    f11 = xsec[:ngas, t_idx, p_idx]
    return (1.0 - fy)[:, None] * ((1.0 - fx)[:, None] * f00 + fx[:, None] * f10) + fy[
        :, None
    ] * ((1.0 - fx)[:, None] * f01 + fx[:, None] * f11)


def _tau_loop_jax(density, sigma, path):
    """Sequential in-place accumulation via ``lax.fori_loop``."""
    import jax.numpy as jnp
    from jax import lax

    def body(layer, tau):
        contribution = jnp.sum(path[layer][:, None] * density[:, None] * sigma, axis=0)
        return tau.at[layer].add(contribution)

    return lax.fori_loop(0, density.shape[0], body, jnp.zeros_like(sigma))


def forward_jax(theta, loop: bool = False, arrays: dict | None = None):
    """JAX port of :func:`forward_np` (pure function of ``theta``).

    Passing ``arrays`` (see :func:`_jax_arrays`) hoists the grid constants out
    of the traced function, which matters for timing; otherwise they are built
    inside the trace, which is correct but re-uploads them on every call.
    """
    import jax.numpy as jnp

    if arrays is None:
        arrays = _jax_arrays()
    ngas = int(theta.shape[0]) - 3
    temperature = guillot_temperature_jax(
        arrays["pressure"], theta[ngas], theta[ngas + 1], theta[ngas + 2]
    )
    sigma_gas = interp_xsec_jax(
        temperature, jnp.log10(arrays["pressure"]), arrays, ngas
    )
    sigma_mix = jnp.einsum("i,ilw->lw", 10.0 ** theta[:ngas], sigma_gas)
    density = arrays["pressure"] / (K_B * temperature)
    if loop:
        tau = _tau_loop_jax(density, sigma_mix, arrays["path"])
    else:
        tau = jnp.einsum("lk,k,kw->lw", arrays["path"], density, sigma_mix)
    trans = jnp.exp(-tau)
    integral = jnp.sum(
        (R_PLANET + arrays["z_center"])[:, None]
        * (1.0 - trans)
        * arrays["dz"][:, None]
        * 2.0,
        axis=0,
    )
    return (R_PLANET**2 + integral) / R_STAR**2


def chisq_jax(theta, loop: bool = False, arrays: dict | None = None, data=None):
    """Chi-squared of the JAX model against the synthetic data."""
    import jax.numpy as jnp

    if arrays is None:
        arrays = _jax_arrays()
    target = arrays["data"] if data is None else data
    residual = (target - forward_jax(theta, loop=loop, arrays=arrays)) / ERR
    return jnp.sum(residual * residual)


def jacobian_jax(theta) -> np.ndarray:
    """Return ``dS/dtheta`` as ``(n_parameters, n_wavenumbers)``.

    ``jax.jacfwd`` returns ``(n_outputs, n_inputs)``; the transpose keeps the
    same orientation as the numpy finite-difference Jacobian.
    """
    jax = _jax()
    import jax.numpy as jnp

    arrays = _jax_arrays()  # built outside the trace on purpose
    traced = jax.jit(jax.jacfwd(lambda t: forward_jax(t, arrays=arrays)))
    return np.asarray(traced(jnp.asarray(theta))).T


def saturation_gate_jax(x, decision_x, limit: float = 10.0):
    """JAX port of :func:`saturation_gate_np`.

    ``decision_x`` must be a concrete (untraced) value: a Python ``if`` on a
    traced value raises ``TracerBoolConversionError`` under ``jax.grad``.
    """
    import jax.numpy as jnp

    saturated = bool(9.999 * decision_x > limit)
    tau = 9.999 * x
    return jnp.where(saturated, tau, tau + 5.0 * x)


# --------------------------------------------------------------------------
# Diagnostics: gradient checks and benchmark
# --------------------------------------------------------------------------
def device_report() -> str:
    """Return a one-line description of the available devices."""
    lines = []
    try:
        import torch

        lines.append(f"torch cuda={torch.cuda.is_available()}")
    except ImportError:
        lines.append("torch missing")
    try:
        jax = _jax()

        lines.append(f"jax backend={jax.default_backend()} devices={jax.devices()}")
    except ImportError:
        lines.append("jax missing")
    return " | ".join(lines)


def gradient_report() -> dict:
    """Compare FD, torch and JAX gradients of the chi-squared."""
    torch = _torch()
    jax = _jax()
    import jax.numpy as jnp

    fd = finite_difference_grad(chisq_np, THETA0)

    theta_t = torch.tensor(THETA0, dtype=torch.float64, requires_grad=True)
    chisq_torch(theta_t).backward()
    torch_grad = theta_t.grad.detach().cpu().numpy()

    jax_grad = np.asarray(
        jax.jit(jax.grad(lambda t: chisq_jax(t, arrays=_jax_arrays())))(
            jnp.asarray(THETA0)
        )
    )
    return {"fd": fd, "torch": torch_grad, "jax": jax_grad}


def jacobian_report() -> dict:
    """Compare FD and autodiff Jacobians of the spectrum ``dS/dtheta``."""
    torch = _torch()

    ref = forward_np(THETA0)
    fd = np.empty((THETA0.size, ref.size))
    for i in range(THETA0.size):
        step = 1e-6 * max(1.0, abs(THETA0[i]))
        up = THETA0.copy()
        down = THETA0.copy()
        up[i] += step
        down[i] -= step
        fd[i] = (forward_np(up) - forward_np(down)) / (2.0 * step)

    theta_t = torch.tensor(THETA0, dtype=torch.float64, requires_grad=True)
    torch_jac = (
        torch.autograd.functional.jacobian(
            lambda t: forward_torch(t), theta_t, vectorize=False
        )
        .detach()
        .cpu()
        .numpy()
    )
    # torch returns (n_outputs, n_params); the numpy/JAX convention here is
    # (n_params, n_wavenumbers).
    torch_jac = torch_jac.T
    jax_jac = jacobian_jax(THETA0)
    return {"fd": fd, "torch": torch_jac, "jax": jax_jac}


def _synchronise(torch_module) -> None:
    """Wait for pending GPU work so that timings include it."""
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def _time_call(torch_module, func, repeats: int) -> float:
    """Return the fastest wall-clock time of ``func`` in seconds.

    The minimum over repeats is used rather than the mean: a laptop GPU shared
    with the display produces occasional slow calls, and the minimum is the
    reproducible part.
    """
    func()
    _synchronise(torch_module)
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        _synchronise(torch_module)
        best = min(best, time.perf_counter() - start)
    return best


def gradient_cost(ngas: int, repeats: int = 5, device: str | None = None) -> dict:
    """Measure what one gradient costs, by finite differences and by autograd.

    Both measurements run in torch on the same device, so the comparison is
    like for like. Finite differences need ``2 * n_params`` forward passes;
    autograd needs one forward plus one backward no matter how many parameters
    there are. That is the entire argument for making the model differentiable.

    Args:
        ngas: number of gases in the model (``n_params = ngas + 3``).
        repeats: timing repetitions.
        device: torch device string; defaults to CUDA when available.

    Returns:
        Dictionary with the parameter count and the timings in milliseconds.
    """
    import torch

    torch_module = _torch()
    device = device or ("cuda" if torch_module.cuda.is_available() else "cpu")
    arrays = _torch_arrays(device)
    theta0, _, spectrum = build_dataset(ngas)
    data = torch.as_tensor(spectrum, dtype=torch.float64, device=arrays["data"].device)
    theta = torch.tensor(
        theta0, dtype=torch.float64, requires_grad=True, device=data.device
    )
    step_rel = 1e-6

    def forward():
        with torch.no_grad():
            forward_torch(theta, arrays=arrays)

    def autograd_grad():
        theta.grad = None
        chisq_torch(theta, arrays=arrays, data=data).backward()

    def finite_difference_grad():
        for i in range(theta.numel()):
            step = step_rel * max(1.0, abs(float(theta0[i])))
            up = theta.detach().clone()
            down = theta.detach().clone()
            up[i] += step
            down[i] -= step
            with torch.no_grad():
                high = chisq_torch(up, arrays=arrays, data=data)
                low = chisq_torch(down, arrays=arrays, data=data)
                (high - low) / (2.0 * step)

    forward_ms = 1e3 * _time_call(torch_module, forward, repeats)
    autograd_ms = 1e3 * _time_call(torch_module, autograd_grad, repeats)
    finite_difference_ms = 1e3 * _time_call(
        torch_module, finite_difference_grad, max(1, repeats // 2)
    )
    return {
        "n_params": int(theta.numel()),
        "forward_ms": forward_ms,
        "autograd_ms": autograd_ms,
        "finite_difference_ms": finite_difference_ms,
        "fd_evaluations": 2 * int(theta.numel()),
        "speedup": finite_difference_ms / autograd_ms,
        "device": device,
    }


def scaling_report(
    ngas_list: tuple[int, ...] = (1, 2, 3, 4, 6, 8),
    repeats: int = 5,
    device: str | None = None,
) -> list[dict]:
    """Measure :func:`gradient_cost` for models of increasing size."""
    return [gradient_cost(ngas, repeats=repeats, device=device) for ngas in ngas_list]


def parameter_bounds(ngas: int) -> tuple[np.ndarray, np.ndarray]:
    """Return generous physical bounds for a theta vector.

    A differentiable model still needs the same bounds a real retrieval uses.
    Without them L-BFGS walks down the nearly flat direction, the mixing ratios
    overflow and the fit returns NaN -- which is worth knowing.

    Args:
        ngas: number of gases in the model.

    Returns:
        Lower and upper bound arrays matching a theta vector.
    """
    lower = np.concatenate([np.full(ngas, -10.0), [300.0, 0.01, 1.0e-6]])
    upper = np.concatenate([np.full(ngas, -1.0), [3000.0, 0.99, 1.0e-1]])
    return lower, upper


def optimizer_comparison(theta_start: np.ndarray, maxiter: int = 400) -> dict:
    """Compare a gradient-based fit with a gradient-free search.

    L-BFGS uses autograd; Nelder-Mead uses function values only. Both minimise
    the same bounded chi-squared on the CPU, so the wall-clock comparison is
    fair. The robust number is the evaluation count expressed in forward passes:
    one autograd gradient costs roughly two forward passes.

    Args:
        theta_start: starting point for both optimisers.
        maxiter: iteration cap for the gradient-free search.

    Returns:
        Dictionary with evaluation counts, timings, fits and chi-squared values.
    """
    import torch
    from scipy.optimize import minimize

    ngas = int(np.size(theta_start)) - 3
    _, _, spectrum = build_dataset(ngas)
    lower, upper = parameter_bounds(ngas)
    lower_t = torch.as_tensor(lower, dtype=torch.float64)
    upper_t = torch.as_tensor(upper, dtype=torch.float64)
    arrays = _torch_arrays("cpu")
    theta = torch.tensor(
        np.clip(np.asarray(theta_start, dtype=np.float64), lower, upper),
        dtype=torch.float64,
        requires_grad=True,
    )
    data = torch.as_tensor(spectrum, dtype=torch.float64)
    optimizer = torch.optim.LBFGS(
        [theta], lr=1.0, max_iter=200, line_search_fn="strong_wolfe"
    )
    history: list[float] = []

    def closure():
        # Project first: L-BFGS steps can leave the box, and an out-of-bounds
        # mixing ratio overflows before the clamp at the end of the previous
        # call could help.
        with torch.no_grad():
            theta.clamp_(lower_t, upper_t)
        optimizer.zero_grad()
        loss = chisq_torch(theta, arrays=arrays, data=data)
        loss.backward()
        history.append(float(loss.detach()))
        return loss

    start = time.perf_counter()
    optimizer.step(closure)
    lbfgs_seconds = time.perf_counter() - start
    lbfgs_fit = theta.detach().numpy()

    def gradient_free_objective(values):
        return chisq_np(np.clip(values, lower, upper), data=spectrum)

    start = time.perf_counter()
    gradient_free = minimize(
        gradient_free_objective,
        np.asarray(theta_start, dtype=np.float64),
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-9, "fatol": 1e-12},
    )
    gradient_free_seconds = time.perf_counter() - start
    gradient_free_fit = np.clip(gradient_free.x, lower, upper)

    return {
        "lbfgs_evaluations": len(history),
        "lbfgs_forward_equivalents": 2 * len(history),
        "lbfgs_seconds": lbfgs_seconds,
        "lbfgs_chisq": float(chisq_np(lbfgs_fit, data=spectrum)),
        "lbfgs_fit": lbfgs_fit,
        "lbfgs_history": history,
        "gradient_free_evaluations": int(gradient_free.nfev),
        "gradient_free_seconds": gradient_free_seconds,
        "gradient_free_chisq": float(chisq_np(gradient_free_fit, data=spectrum)),
        "gradient_free_fit": gradient_free_fit,
        "gradient_free_success": bool(gradient_free.success),
    }


def framework_report(batch_size: int = 256, repeats: int = 5) -> dict:
    """Compare PyTorch and JAX on the same model, device and batch.

    Measures what the framework choice actually changes: per-call cost, whether
    the model can be batched with ``vmap``, batched throughput, and the cost of
    the first jitted call.

    Args:
        batch_size: number of parameter sets in the batched measurements. 256
            fits a 4 GB card; 1024 exhausts it during the batched *gradient*
            (the reverse-mode tape of a 40-layer model is ~100 MB per saved
            activation), which is the report's memory caveat in miniature.
        repeats: timing repetitions; the fastest time is reported.

    Returns:
        Dictionary of timings in milliseconds and capability flags.
    """
    import torch

    torch_module = _torch()
    jax = _jax()
    import jax.numpy as jnp

    device = "cuda" if torch_module.cuda.is_available() else "cpu"
    jax_arrays = _jax_arrays()
    torch_arrays = _torch_arrays(device)
    theta0, _, _ = build_dataset(1)
    rng = np.random.default_rng(0)
    batch = theta0[None, :] * (
        1.0 + 1e-2 * rng.standard_normal((batch_size, theta0.size))
    )
    batch_jax = jnp.asarray(batch)
    batch_torch = torch.as_tensor(batch, dtype=torch.float64, device=device)
    theta_jax = jnp.asarray(theta0)
    theta_grad = torch.tensor(
        theta0, dtype=torch.float64, requires_grad=True, device=device
    )
    theta_plain = torch.tensor(theta0, dtype=torch.float64, device=device)

    def fastest(func, count: int = repeats) -> float:
        """Fastest of ``count`` calls, in milliseconds."""
        func()
        _synchronise(torch_module)
        best = float("inf")
        for _ in range(count):
            start = time.perf_counter()
            func()
            _synchronise(torch_module)
            best = min(best, time.perf_counter() - start)
        return 1e3 * best

    def torch_forward() -> None:
        with torch.no_grad():
            forward_torch(theta_plain, arrays=torch_arrays)

    def torch_forward_backward() -> None:
        theta_grad.grad = None
        chisq_torch(theta_grad, arrays=torch_arrays).backward()

    def torch_loop() -> None:
        # No vmap for this model, so the only torch batching is a Python loop.
        with torch.no_grad():
            for row in batch_torch:
                forward_torch(row, arrays=torch_arrays)

    jax_forward = jax.jit(lambda t: forward_jax(t, arrays=jax_arrays))
    jax_grad = jax.jit(jax.grad(lambda t: chisq_jax(t, arrays=jax_arrays)))
    jax_vmap_forward = jax.jit(jax.vmap(lambda t: forward_jax(t, arrays=jax_arrays)))
    jax_vmap_grad = jax.jit(
        jax.vmap(jax.grad(lambda t: chisq_jax(t, arrays=jax_arrays)))
    )

    # The exception *is* the measurement: the E_2 primal calls scipy on the host,
    # so the custom autograd.Function has no vmap rule and cannot be batched.
    torch_vmap_error = None
    try:
        torch.vmap(lambda t: forward_torch(t, arrays=torch_arrays))(batch_torch)
    except (RuntimeError, TypeError) as exc:
        torch_vmap_error = f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}"

    report = {
        "device": device,
        "batch_size": batch_size,
        "torch_forward_ms": fastest(torch_forward),
        "torch_forward_backward_ms": fastest(torch_forward_backward),
        "torch_loop_batch_ms": fastest(torch_loop, count=2),
        "torch_vmap_error": torch_vmap_error,
        "jax_forward_ms": fastest(lambda: jax_forward(theta_jax).block_until_ready()),
        "jax_forward_grad_ms": fastest(lambda: jax_grad(theta_jax).block_until_ready()),
        "jax_vmap_forward_ms": fastest(
            lambda: jax_vmap_forward(batch_jax).block_until_ready()
        ),
        "jax_vmap_grad_ms": fastest(
            lambda: jax_vmap_grad(batch_jax).block_until_ready()
        ),
    }

    start = time.perf_counter()
    jax.jit(lambda t: forward_jax(t, arrays=jax_arrays))(theta_jax).block_until_ready()
    report["jax_forward_compile_ms"] = 1e3 * (time.perf_counter() - start)
    start = time.perf_counter()
    jax.jit(jax.grad(lambda t: chisq_jax(t, arrays=jax_arrays)))(
        theta_jax
    ).block_until_ready()
    report["jax_grad_compile_ms"] = 1e3 * (time.perf_counter() - start)
    report["jax_per_spectrum_us"] = 1e3 * report["jax_vmap_forward_ms"] / batch_size
    report["torch_per_spectrum_us"] = 1e3 * report["torch_loop_batch_ms"] / batch_size
    return report


def saturation_report() -> dict:
    """Quantify the discontinuity introduced by the early-out shortcut."""
    torch = _torch()
    jax = _jax()
    import jax.numpy as jnp

    fd = (saturation_gate_np(1.0 + 1e-3) - saturation_gate_np(1.0 - 1e-3)) / 2e-3
    x_t = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    torch_grad = torch.autograd.grad(saturation_gate_torch(x_t), x_t)[0].item()
    jax_grad = float(jax.grad(lambda v: saturation_gate_jax(v, 1.0))(jnp.asarray(1.0)))
    x_t2 = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    detached = torch.autograd.grad(
        saturation_gate_torch(x_t2, decision_x=x_t2.detach()), x_t2
    )[0].item()
    return {
        "finite_difference": fd,
        "autodiff": torch_grad,
        "jax_autodiff": jax_grad,
        "detached_decision": detached,
    }


def benchmark(repeats: int = 20) -> list[tuple[str, float]]:
    """Time forwards and gradients for numpy, torch and JAX."""
    torch = _torch()
    jax = _jax()
    import jax.numpy as jnp

    results: list[tuple[str, float]] = []

    def timed(label: str, func, repeats: int = repeats) -> None:
        func()
        start = time.perf_counter()
        for _ in range(repeats):
            func()
        results.append((label, (time.perf_counter() - start) / repeats * 1e3))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    theta_t = torch.tensor(
        THETA0, dtype=torch.float64, requires_grad=True, device=device
    )
    theta_j = jnp.asarray(THETA0)

    timed("numpy  forward (einsum)", lambda: forward_np(THETA0))
    timed("numpy  forward (layer loop)", lambda: forward_np(THETA0, loop=True))
    timed(f"torch  forward ({device}, einsum)", lambda: forward_torch(theta_t))
    timed(
        f"torch  forward ({device}, layer loop)",
        lambda: forward_torch(theta_t, loop=True),
    )

    def torch_grad_step():
        theta_t.grad = None
        chisq_torch(theta_t).backward()

    timed(f"torch  forward+backward ({device})", torch_grad_step)

    jax_arrays = _jax_arrays()  # built once, outside the traces
    jax_forward = jax.jit(lambda t: forward_jax(t, arrays=jax_arrays))
    jax_grad = jax.jit(jax.grad(lambda t: chisq_jax(t, arrays=jax_arrays)))
    jax_loop = jax.jit(lambda t: forward_jax(t, loop=True, arrays=jax_arrays))

    jax_forward(theta_j).block_until_ready()
    timed("jax    forward (jit)", lambda: jax_forward(theta_j).block_until_ready())
    jax_grad(theta_j).block_until_ready()
    timed("jax    forward+grad (jit)", lambda: jax_grad(theta_j).block_until_ready())
    jax_loop(theta_j).block_until_ready()
    timed("jax    forward loop (jit)", lambda: jax_loop(theta_j).block_until_ready())
    return results


def main() -> None:
    """Print the gradient checks and benchmark table."""
    print("=" * 78)
    print("TauREx3 differentiability prototype")
    print("=" * 78)
    print(f"devices      : {device_report()}")
    print(f"layers/grid  : {NLAYERS} / {NWN} (wavenumbers)")
    spectrum = forward_np(THETA0)
    print(f"transit depth: {spectrum.min():.3e} .. {spectrum.max():.3e}")

    print("\n-- chi-squared gradient, theta =", ", ".join(THETA_NAMES))
    grads = gradient_report()
    print(f"{'param':>10} {'finite diff':>16} {'torch':>16} {'jax':>16}")
    for i, name in enumerate(THETA_NAMES):
        print(
            f"{name:>10} {grads['fd'][i]:16.9e} {grads['torch'][i]:16.9e}"
            f" {grads['jax'][i]:16.9e}"
        )
    torch_err = np.max(np.abs(grads["torch"] - grads["fd"]) / np.abs(grads["fd"]))
    jax_err = np.max(np.abs(grads["jax"] - grads["fd"]) / np.abs(grads["fd"]))
    print(
        f"max relative error vs finite diff: torch {torch_err:.2e}, jax {jax_err:.2e}"
    )

    print("\n-- spectrum Jacobian dS/dtheta")
    jac = jacobian_report()
    scale = np.max(np.abs(jac["fd"]))
    print(f"max |FD|          = {scale:.3e}")
    print(f"max |torch - FD|  = {np.max(np.abs(jac['torch'] - jac['fd'])):.3e}")
    print(f"max |jax   - FD|  = {np.max(np.abs(jac['jax'] - jac['fd'])):.3e}")

    print("\n-- what one gradient costs (torch, same device)")
    cost = gradient_cost(NGAS)
    print(f"n_params            = {cost['n_params']} (device {cost['device']})")
    print(f"forward             = {cost['forward_ms']:8.3f} ms")
    print(f"autograd gradient   = {cost['autograd_ms']:8.3f} ms")
    print(
        f"finite differences  = {cost['finite_difference_ms']:8.3f} ms"
        f"  ({cost['fd_evaluations']} forward passes)"
    )
    print(f"autograd speed-up   = {cost['speedup']:8.2f}x")

    print("\n-- scaling with the parameter count (torch)")
    print(
        f"{'n_params':>9} {'forward':>9} {'autograd':>9} {'finite diff':>12}"
        f" {'ratio':>7}"
    )
    for row in scaling_report():
        print(
            f"{row['n_params']:>9} {row['forward_ms']:>9.3f} {row['autograd_ms']:>9.3f}"
            f" {row['finite_difference_ms']:>12.3f} {row['speedup']:>7.2f}"
        )

    print("\n-- fit cost: gradient-based vs gradient-free (CPU)")
    fits = optimizer_comparison(THETA0)
    print(
        f"L-BFGS      : {fits['lbfgs_evaluations']} gradients,"
        f" {fits['lbfgs_seconds'] * 1e3:.1f} ms, chi2={fits['lbfgs_chisq']:.3g}"
    )
    print(
        f"Nelder-Mead : {fits['gradient_free_evaluations']} evaluations,"
        f" {fits['gradient_free_seconds'] * 1e3:.1f} ms,"
        f" chi2={fits['gradient_free_chisq']:.3g}"
    )

    print("\n-- framework comparison (batch of 1024 parameter sets)")
    framework = framework_report(batch_size=1024)
    rows = (
        ("single forward", framework["torch_forward_ms"], framework["jax_forward_ms"]),
        (
            "single forward+backward",
            framework["torch_forward_backward_ms"],
            framework["jax_forward_grad_ms"],
        ),
        (
            "batched forward x1024",
            framework["torch_loop_batch_ms"],
            framework["jax_vmap_forward_ms"],
        ),
        ("first call (compile)", 0.0, framework["jax_forward_compile_ms"]),
    )
    print(f"{'':<26}{'PyTorch':>12}{'JAX':>12}")
    for label, torch_ms, jax_ms in rows:
        print(f"{label:<26}{torch_ms:>9.2f} ms{jax_ms:>9.2f} ms")
    print(
        f"{'batched gradient x1024':<26}{'unavailable':>12}"
        f"{framework['jax_vmap_grad_ms']:>9.2f} ms"
    )
    print(f"torch vmap : {framework['torch_vmap_error']}")

    print("\n-- E_2(x) special function")
    torch = _torch()
    jax = _jax()
    import jax.numpy as jnp

    x = torch.linspace(0.1, 5.0, 5, dtype=torch.float64, requires_grad=True)
    torch_val = expn2_torch(x).detach().numpy()
    scipy_val = scipy_expn(2, x.detach().numpy())
    print(f"torch vs scipy max abs err: {np.max(np.abs(torch_val - scipy_val)):.3e}")
    gradcheck = torch.autograd.gradcheck(expn2_torch, (x,))
    print(f"torch gradcheck            : {gradcheck}")

    # Argument range the Guillot profile actually produces.
    gamma_tau = np.logspace(-9, 4.5, 200)
    jax_val = np.asarray(jax.jit(expn2_jax)(jnp.asarray(gamma_tau)))
    print(
        f"jax vs scipy max rel err   : "
        f"{np.max(np.abs(jax_val - scipy_expn(2, gamma_tau))):.3e}"
    )
    start = time.perf_counter()
    jax.jit(expn2_jax)(jnp.asarray(gamma_tau)).block_until_ready()
    print(f"jax E_2 on the real range  : {1e3 * (time.perf_counter() - start):.3f} ms")

    print("\n-- saturation early-out (report obstacle 6)")
    sat = saturation_report()
    for key, value in sat.items():
        print(f"{key:>20}: {value:+.6f}")

    print("\n-- timing (ms per call)")
    for label, millis in benchmark():
        print(f"{label:<28} {millis:8.3f}")


if __name__ == "__main__":
    main()

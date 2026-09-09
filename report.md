# Making TauREx3 Differentiable — Deep Analysis & Implementation Plan

- **Branch:** `prototype_differentiability` (currently identical to `main`, commit `a1cd38c`)
- **Date:** 2026-09-09
- **Scope:** how hard/efficient it is to make the forward model + optimizer differentiable, what gradients are worth, and a JAX vs PyTorch comparison with concrete implementation steps.

---

## 1. Executive summary

TauREx3's **forward mathematics is already differentiable** — it is mostly `einsum`, `exp`, `log`, `sqrt` and polynomial algebra. The barriers to autodiff are **structural, not mathematical**:

1. Fitting parameters live in Python floats behind `@fitparam` property getters/setters (side-effect mutation), not in traced tensors.
2. Opacity lookup uses `searchsorted`-style **index selection** (piecewise-constant, zero gradient) and `scipy.interpolate.interp1d`.
3. Hot loops are compiled with **numba** (`nopython=True`) and contain **data-dependent control flow** (`if tau[layer].min() > 10: continue`).
4. The `Optimizer.chisq_trans` swallows errors into `np.nan`.

**Recommendation:** prototype in **PyTorch** first (fits the existing OOP/`Fittable` design, incremental adoption), then decide whether to invest in a **JAX** port if HMC/NUTS at scale or `vmap`-batched training of neural emulators becomes the goal. A working `∂χ²/∂θ` for a single-contribution transmission model is a 2–4 week task in PyTorch and a larger (but ultimately more powerful) rewrite in JAX.

---

## 2. The current compute graph

```
fitting parameters (@fitparam: T_surface, mix_ratio, kappa_ir, ...)
      │  (property getters/setters, Python floats)
      ▼
PressureProfile.profile ──▶ per-layer P (Pa)
TemperatureProfile.profile ─▶ per-layer T (K)      e.g. Guillot2010, NPoint, Isothermal
Chemistry / Gas.mixProfile ─▶ per-layer mixing ratios   e.g. PowerGas, ConstantGas
      │
      ▼
Contribution.prepare(model, wngrid)   ──▶ sigma_xsec via opacity interpolation over (T, P)
      │
      ▼
TransmissionModel.path_integral(wngrid)
      │   Python loop over layers:
      │     - path length (sqrt of radius differences)
      │     - if tau[layer].min() > 10: continue   ◀── data-dependent control flow
      │     - contrib.contribute(...) → contribute_tau/ktau/cia kernels (numba/numpy)
      │         tau[layer, :] += einsum("ki,k,k->i", sigma, path, density)
      ▼
compute_absorption(tau, dz)
      │   np.exp(-tau, out=tau)
      │   integral = sum((R_planet + alt) * (1 - tau) * dz * 2, axis=0)
      ▼
Binner.bin_model(model_output)  ──▶ binned spectrum at observed grid
      ▼
Optimizer.chisq_trans → log_likelihood   (try/except → np.nan on invalid model)
```

### 2.1 Key modules and their role

| Module                            | File                                                         | Role                                                                           |
| --------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| `ForwardModel`                    | `src/taurex/model/model.py`                                  | Base; `contribution_list`, `fittingParameters`, `derivedParameters`, `model()` |
| `TransmissionModel`               | `src/taurex/model/transmission.py`                           | `path_integral`, `compute_path_length(_old)`, `compute_absorption`             |
| `AbsorptionContribution`          | `src/taurex/contributions/absorption.py`                     | molecular absorption; k-table and non-k-table paths                            |
| `CIA` contribution                | `src/taurex/contributions/cia.py`                            | collision-induced absorption (`density²`)                                      |
| `Rayleigh`, `LeeMie`, etc.        | `src/taurex/contributions/`                                  | scattering / clouds                                                            |
| `Opacity` / `KTable`              | `src/taurex/opacity/opacity.py`, `opacity/ktables/ktable.py` | cross-section computation + interpolation                                      |
| `InterpolatingOpacity`            | `src/taurex/opacity/interpolateopacity.py`                   | bilinear (T,P) interpolation via index lookup                                  |
| `util/math.py` (+`math_numba.py`) | `src/taurex/util/`                                           | `interp_lin/exp`, `intepr_bilin` (numba-accelerated)                           |
| `Fittable` / `fitparam`           | `src/taurex/data/fittable.py`                                | parameter declaration (getter/setter properties)                               |
| `Optimizer` / `FitParam`          | `src/taurex/optimizer/optimizer.py`                          | `update_model`, `log_likelihood`, `chisq_trans`                                |
| Profiles                          | `src/taurex/data/profiles/{temperature,pressure,chemistry}/` | `NPoint`, `Guillot2010`, `PowerGas`, `SimplePressureProfile`, ...              |
| `Binner`                          | `src/taurex/binning/binner.py`                               | rebin native spectrum onto observed grid                                       |

---

## 3. What you gain from differentiability (the "why")

The current optimizer stack is **gradient-free**: `nestle`, `pymultinest`, `polychord` (all nested sampling). Nested sampling is excellent for posteriors with a handful of dimensions but its cost grows **exponentially** with the number of parameters and it wastes information (it only needs function values, never slope). Differentiability unlocks a different class of algorithms:

### 3.1 Direct gains, ordered by immediate usefulness

1. **Fast MAP / point estimates (gradient descent).**
   `∇θ χ²(θ)` lets you run L-BFGS, Adam or Newton-CG to the maximum a-posteriori point in tens of iterations instead of tens of thousands of likelihood evaluations. This is the cheapest win and the natural first deliverable.

2. **HMC / NUTS sampling.**
   Hamiltonian Monte Carlo (via NumPyro/BlackJAX for JAX, or Pyro/Hamiltorch for PyTorch) uses `∇θ log L` to propose states. It scales far better than nested sampling to high-dimensional parameter spaces (e.g. NPoint profiles with many temperature knots, or chemistry with many gases), and returns posterior samples directly.

3. **Fisher information matrix & posterior covariance.**
   With the Jacobian $J = \partial\,\text{model}/\partial\theta$ (one backward pass for all parameters) the Fisher matrix is
   $$F = J^\top \Sigma^{-1} J, \qquad \text{Cov}(\theta) \approx F^{-1}.$$
   This gives the **Laplace approximation** of the posterior and error bars on every parameter — without any sampling — plus error propagation to derived parameters via the chain rule.

4. **Optimal experimental design.**
   Design metrics (D-optimality, mutual information, KL) are functions of $F$. With gradients w.r.t. the _data/observation model_ you can decide which wavelengths, resolutions, or instruments add the most information.

5. **Adjoint sensitivity analysis.**
   Reverse-mode AD computes $\partial\,\text{spectrum}/\partial\theta$ for **all** parameters in a single backward pass (memory cost only), letting you cheaply map "which parameters drive which spectral features".

6. **Neural emulators / surrogates.**
   A differentiable forward model is the ideal ground truth for training a neural network that reproduces the spectrum from parameters:

   - gradient-based training data generation,
   - **physics-informed / gradient-matching losses** (match $f$ _and_ $\partial f/\partial\theta$),
   - composing the emulator + likelihood into a fully differentiable pipeline for amortized inference (normalizing flows) that runs in milliseconds.

7. **Variational inference (ADVI).**
   Fit a parameterized posterior $q_\phi(\theta)$ by gradient descent on the ELBO; requires `∇θ log L` and reparameterized sampling.

8. **Derived-parameter gradients for free.**
   Derived parameters (`avg_T`, etc.) and error bars on them come from the same graph.

### 3.2 What it costs

- **Reverse-mode memory:** the layer loop builds an intermediate `tau` array of shape `(nlayers, nwavenumber)` and a cross-section `sigma_xsec` per contribution. Storing activations for `nlayers` (100+) layers is manageable (a few GB worst case), but naive per-layer Python loops will make the tape long.
- **Forward overhead:** a naive differentiable forward pass is ~2–5× slower than the numba path; vectorizing the layer loop (`vmap` / batched einsum) recovers most of this and can even beat the Python loop.
- **Non-smoothness:** index-based interpolation is differentiable a.e. but has flat gradient w.r.t. temperature/pressure _at the grid selection step_ (see §4.2). Gradients are valid almost everywhere; HMC still works well in practice.

---

## 4. Inventory: what is NOT differentiable today

### 4.1 Component-by-component

| #   | Obstacle                                                                                  | Location                                                                                                                              | Nature                                                      |
| --- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| 1   | `@fitparam` params are Python floats mutated by setter side-effects; `FitParam.fget/fset` | `src/taurex/data/fittable.py`, `optimizer/optimizer.py::update_model`                                                                 | **Structural** (breaks tracing)                             |
| 2   | `np.searchsorted` / `find_closest_pair` index lookup for (T,P) opacity grid               | `opacity/interpolateopacity.py`, `util/math.py`                                                                                       | Zero-gradient indices                                       |
| 3   | `scipy.interpolate.interp1d` (k-table rebinning)                                          | `opacity/ktables/ktable.py`                                                                                                           | External C, untraceable                                     |
| 4   | `scipy.special.expn` (exponential integral in Guillot2010)                                | `data/profiles/temperature/guillot.py`                                                                                                | Special function, needs AD impl                             |
| 5   | numba `nopython=True` kernels                                                             | `contribution.py::contribute_tau_numba`, `absorption.py::contribute_ktau_numba`, `cia.py::contribute_cia_numba`, `util/math_numba.py` | Untraceable JIT                                             |
| 6   | Data-dependent control flow `if tau[layer].min() > 10: continue`                          | `model/transmission.py::path_integral`                                                                                                | Non-differentiable branch                                   |
| 7   | Python `for layer in range(nlayers)` with in-place `tau[layer,:] += ...`                  | `model/transmission.py::path_integral`                                                                                                | In-place mutation on traced arrays                          |
| 8   | `np.where`, clipping, `GlobalCache` / `OpacityCache` lookups                              | `opacity/opacity.py`, `cache/`                                                                                                        | Untraceable / non-diff                                      |
| 9   | `try/except InvalidModelException → np.nan` in `chisq_trans`                              | `optimizer/optimizer.py`                                                                                                              | Kills gradient silently                                     |
| 10  | Log-space transforms `math.log10`, prior sampling                                         | `optimizer/optimizer.py`, `core/priors.py`                                                                                            | Python scalars, non-diff transforms                         |
| 11  | In-place `np.exp(-tau, out=tau)`                                                          | `model/transmission.py::compute_absorption`                                                                                           | In-place on traced array                                    |
| 12  | Binner rebinning (summing into bins)                                                      | `binning/binner.py`                                                                                                                   | Fine if rewritten with `segment_sum` / `bincount`-style ops |

### 4.2 The two genuinely hard problems

**A. Opacity interpolation (index lookup).**
`InterpolatingOpacity.find_closest_index` uses `find_closest_pair` (a searchsorted) to pick `(t_min, t_max, p_min, p_max)`. The interpolation arithmetic (`interp_lin_only`, `intepr_bilin`) is differentiable, but the _indices_ are integer-valued functions of `(T, P)` → zero gradient and non-smooth. Standard fix in both frameworks:

- Compute indices with `searchsorted` and **detach them** (`torch.detach` / `jax.lax.stop_gradient`) so gradients flow only through the interpolation weights and the grid values.
- Use `interp1d`-equivalent **linear interpolation as a gather + weighted sum**: `y = x0*(1-f) + x1*f` with `f` the fractional position.
- For k-tables, replace `scipy.interpolate.interp1d` with `torch`/`jax` linear interp over the wavenumber axis.

**B. The layer loop + numba kernels + early-out.**
The `if tau[layer].min() > 10: continue` is a numerical shortcut (optical depth saturated). Options:

- Drop it (correctness first, cost only in deep layers), or
- Replace with a **masked reduction** (`jnp.where(tau_min > 10, 0, contrib)`) — differentiable but discontinuous at the threshold, or
- Keep it as a _fixed_ decision taken on a detached copy (no gradient path through the branch).

The `for layer` loop must become a batched operation (`jax.vmap` / `torch.vmap` / batched einsum over a `(nlayers, nlayers, ngrid)` structure), because reverse-mode AD does not play well with 100 sequential in-place `+=` updates.

---

## 5. JAX implementation plan

JAX requires **functional purity**: no side effects, no in-place mutation, parameters passed as an explicit PyTree. This is the largest conceptual change.

### 5.1 Parameters → PyTree

- Define a dataclass / `pytree` structure of all fitting parameters, e.g.
  ```python
  @chex.dataclass
  class ModelParams:
      T_surface: jnp.ndarray   # or float
      T_top: jnp.ndarray
      mix_ratios: dict[str, jnp.ndarray]
      ...
  ```
- Each profile class gains a **pure function** `profile(params) -> jnp.ndarray` instead of a property reading `self.T_surface`.
- The `@fitparam` decorator can remain for **metadata** (names, bounds, priors) but must stop being the _storage_ mechanism during tracing. A thin adapter maps `FitParam` ↔ PyTree leaf.

### 5.2 Forward model as a pure function

- `path_integral(params, wngrid) -> tau` with `jax.vmap` over layers; replace `if tau.min() > 10: continue` with `jnp.where` masks or `lax.cond`.
- Replace numba kernels with `jnp.einsum` under `jax.jit` (near 1:1 translation of the existing numpy kernels).
- `compute_absorption`: `tau = jnp.exp(-tau)` (no `out=`), `jnp.sum` over layers.
- Opacity: `jnp.interp` for k-tables; custom bilinear using `jnp.searchsorted` + `lax.stop_gradient(indices)` + linear weight interpolation; drop `scipy.interpolate`.
- `Guillot2010`: use `jax.scipy.special.expn` (available in JAX) or a series expansion.
- Binner: `jax.ops.segment_sum` (or `jnp.bincount`-style) for rebinning.

### 5.3 Optimizer / likelihood

- `chisq(params) = ((data - model(params)) / std)² .sum()` as a pure function; remove the `try/except` nan swallow (return `+inf`/`nan` explicitly, or clip).
- `grad = jax.grad(chisq)`; `hess = jax.hessian(chisq)`; `jax.jit` the whole thing; cache compilation per wavenumber grid shape.
- HMC/NUTS: **NumPyro** or **BlackJAX** directly consume the pure function.

### 5.4 Effort & payoff

- **Effort:** high — touches every profile, contribution, opacity and the optimizer; requires functional rewrite of ~15–20 files and a PyTree convention.
- **Payoff:** the highest of the two. `jit + vmap + grad + hessian` compose; GPU/TPU ready; one backward pass for all parameters; first-class ecosystem (NumPyro, BlackJAX, Equinox, Pallas).

### 5.5 JAX pitfalls specific to this codebase

- `jax.jit` recompiles on shape change → pin `nlayers` and `wngrid` shapes, or use `static_argnums`.
- `searchsorted` indices must be passed through `stop_gradient` or computed outside the differentiable graph.
- The `Guillot2010._check_values` raises on invalid parameters → must be replaced with `jnp.where` / `lax.cond` or an infinite-likelihood convention.
- Avoid Python `if` on traced values; use `lax.cond`/`lax.select`.

---

## 6. PyTorch implementation plan

PyTorch's **eager, dynamic-graph** model fits TauREx's OOP design much better and allows **incremental** adoption.

### 6.1 Parameters as tensors

- Keep the `@fitparam` property pattern, but store values as `torch.nn.Parameter` (or plain tensors with `requires_grad=True`) held by each component.
- The setter becomes `self._T_surface = value` where `value` is a tensor; the getter returns the tensor. `FitParam.fget/fset` and `update_model` keep working unchanged (mutation is fine in eager mode).
- `FitParam.value`/`fit_value` currently do `math.log10` on floats → convert to `torch.log10` for traced values.

### 6.2 Forward model

- `path_integral`: keep the Python loop initially (correct but slow), then replace with `torch.vmap`/batched einsum. The `if tau.min() > 10: continue` **works in eager mode** (dynamic graph), but it makes the gradient discontinuous at the threshold — better to remove it or base it on a detached copy.
- Replace `np.einsum` → `torch.einsum`; `np.exp` → `torch.exp` (no `out=` needed; autograd handles in-place only if you call `torch.exp` then assign).
- `compute_absorption`: `tau = torch.exp(-tau)`, `(pradius + ap) * (1 - tau) * dz * 2).sum(0)`.
- numba kernels → vectorized torch ops (drop numba for the differentiable path).
- Opacity interpolation: `torch.searchsorted` + `torch.gather` with `indices.detach()`, weights via linear fraction; or `torch.nn.functional.grid_sample` for the 2-D (T,P) surface.
- k-tables: `torch.nn.functional.interpolate` (1-D) or gather-based linear interp instead of `scipy.interpolate.interp1d`.
- `Guillot2010`: `torch.special` lacks `expn`; implement it with a custom `torch.autograd.Function` (forward via `scipy.special.expn`, backward via the identity $\partial E_n(x)/\partial x = -E_{n-1}(x)$), or approximate.
- Binner: `torch.bincount` / `segment_sum`-style rebinning.

### 6.3 Optimizer / likelihood

- `chisq` returns a 0-dim tensor; `torch.autograd.grad(chisq, params)` or `chisq.backward()` + `.grad`.
- Remove the `try/except → np.nan` (or replace with `torch.where(valid, chi, inf)` so the graph survives).
- HMC/NUTS: **Pyro**, **Hamiltorch**, or `torch.optim.LBFGS` for MAP first.
- `torch.autograd.gradcheck` against finite differences is the correctness gate.

### 6.4 Effort & payoff

- **Effort:** low–medium — mostly mechanical `numpy → torch` translation plus the interpolation replacement; the OOP design survives; you can even keep numba for the non-differentiable "value-only" path and add a torch path in parallel.
- **Payoff:** good, but weaker than JAX on whole-program compilation (`torch.compile` is improving but less battle-tested for this pattern); per-op overhead in the Python layer loop is the main performance risk.

---

## 7. Framework comparison

| Criterion                                       | JAX                                                  | PyTorch                                               |
| ----------------------------------------------- | ---------------------------------------------------- | ----------------------------------------------------- |
| Fit with existing OOP / `@fitparam` design      | **Poor** — must go functional/PyTree                 | **Good** — eager, mutation OK, incremental            |
| Refactor size                                   | Large (purity enforced everywhere)                   | Moderate (mechanical numpy→torch)                     |
| numba / scipy replacement                       | `jit`/`vmap`, `jnp.interp`, `jax.scipy.special.expn` | vectorized ops, `gather`/`grid_sample`, custom `expn` |
| Data-dependent control flow (`if tau.min()>10`) | `lax.cond` / masks required                          | works as-is (non-smooth at threshold)                 |
| `grad` + `jit` + `vmap` + `hessian`             | First-class, composable, fast                        | `torch.func` works; jit less mature                   |
| GPU/TPU & large-scale HMC                       | Excellent (NumPyro / BlackJAX)                       | Good (Pyro / Hamiltorch)                              |
| Compilation model                               | AOT, shape-cached, slow first call                   | Eager (low overhead), `torch.compile` optional        |
| Interpolation gradients                         | `lax.stop_gradient(indices)`                         | `indices.detach()`                                    |
| Memory (reverse-mode tape)                      | Trace + XLA buffers, efficient under `jit`           | Python graph nodes, heavier for 100-layer loops       |
| Effort to first working `∂χ²/∂θ`                | High                                                 | **Low–Medium**                                        |
| Best use case                                   | HMC at scale, batched emulator training, VI          | Prototype, incremental port, mixed numba/torch        |

### 7.1 Decision rule

- **Prototype now, keep the existing codebase shape:** PyTorch.
- **Target HMC/NUTS on 20+ parameters, GPU, or train emulators with `vmap` over thousands of atmospheres:** JAX.
- A hybrid is viable: JAX for the differentiable core (profiles → tau → spectrum), PyTorch for anything else. This is common in astronomy pipelines (e.g. `jax` inside a research workflow that already uses PyTorch for NNs).

---

## 8. Common core work (needed regardless of framework)

These are the changes that both paths share:

1. **A functional parameter abstraction.** A single place (`ModelParams`) that maps fitting-parameter names to leaves; `@fitparam` keeps metadata only.
2. **Vectorized optical-depth kernels.** Rewrite `contribute_tau`, `contribute_ktau`, `contribute_cia` as pure batched einsum functions with **no in-place updates and no data-dependent branches**. Keep numba versions for the value-only fast path.
3. **Differentiable interpolation layer.** One `interp_lin`/`interp_bilin`/`interp1d` replacement using gather + fractional weights with detached indices; replace `scipy.interpolate.interp1d` in `KTable.opacity`.
4. **Special-function replacements.** `scipy.special.expn` (Guillot2010), and any other `scipy.special`/`erf` usages, need framework-native or custom-autograd implementations.
5. **Remove/rework the `try/except → np.nan`** in `chisq_trans` so an invalid model is represented as an explicit large loss (`inf`) rather than a swallowed `nan` (which has undefined gradient).
6. **Binner as a differentiable reduction** (`segment_sum`/`bincount`).
7. **Gradient correctness gate.** Finite-difference / `gradcheck` tests comparing `∇θ χ²` against central differences for a known single-contribution setup.

---

## 9. Recommended roadmap

### Phase 0 — feasibility spike (days)

Pick the smallest differentiable unit: `TransmissionModel` + `AbsorptionContribution` + `Guillot2010` (or `NPoint`) + one `ConstantGas` + one opacity, no clouds, no k-tables. Manually port only `profile → tau → spectrum → χ²`. Validate `∂χ²/∂(T_surface, T_top, log mix_ratio)` against finite differences.

### Phase 1 — framework prototype (weeks)

- **PyTorch:** introduce `torch` tensors behind the existing properties; port `path_integral` + `compute_absorption` + interpolation; get `∂χ²/∂θ` via autograd; add `gradcheck`.
- **JAX (if chosen):** establish the PyTree + pure-function layer; `jax.grad`/`jax.jit`; NumPyro NUTS smoke test.

### Phase 2 — generality (weeks)

- Extend to all contributions (CIA, Rayleigh, Mie, clouds), k-tables, and the `interp_*` family.
- Vectorize the layer loop (`vmap`/`torch.vmap`) and benchmark against the numba path.
- Make the Binner differentiable.

### Phase 3 — algorithm integration

- MAP with L-BFGS/Adam on `χ²`.
- HMC/NUTS sampler alongside the existing `nestle`/`multinest` optimizers (add a `GradientOptimizer`/`HMC` subclass of `Optimizer`).
- Fisher matrix + Laplace posterior + error propagation to derived parameters.

### Phase 4 — advanced (optional)

- Neural emulator trained against the differentiable model (gradient-matching loss).
- Optimal experimental design via the Fisher matrix.

---

## 10. Risks & caveats

- **Non-smooth interpolation gradients** (a.e. differentiability) are acceptable for HMC in practice but can confuse gradient _descent_ at grid boundaries.
- **Memory:** reverse-mode over a 100-layer × (grid ~10⁴) tau requires care; vectorize and avoid materializing full `(nlayers, nlayers, ngrid)` intermediates where possible (the current `einsum` already avoids the `(k, ngrid)` broadcast — preserve that).
- **Compilation cost (JAX):** first-call latency per shape; pin grid sizes or use `static_argnums`.
- **Silent `nan` swallowing** in `chisq_trans` must go, otherwise gradients are silently undefined.
- **Numerical shortcuts** (`tau.min() > 10` early-out, clipping, `np.where` filters) each introduce a kink; keep them but detach where they select indices, or convert to smooth masks.
- **Dual maintenance:** keep the numba numpy path as the fast value-only path and the AD path as an opt-in, so existing retrieval performance is unchanged until the AD path is proven.

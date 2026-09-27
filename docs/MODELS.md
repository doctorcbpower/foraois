# foraois: models and parameter reference

This document lists the exact equations implemented in [`src/foraois/`](../src/foraois) and every config parameter that controls them. For module responsibilities and usage, see [README.md](../README.md). Wavenumbers `k` are in h/Mpc unless stated otherwise; masses in Msun/h; `h = H0/100`.

This file tells us what equations and where they come from. Each module's docstring explains why we have implemented code the way we have (design/architecture choices -- e.g. `pch_trees.py`'s fixed-`dz` vs. PCH08's own adaptive step, `zhang_hui_constrained_trees.py`'s running-maximum construction vs. a Nadler et al. 2023's Appendix A4); [README.md](../README.md#testing)'s Testing section is "what's validated against what." Cite this file for equations/parameters, the relevant module docstring for a design choice's rationale, and the Testing section for validation coverage.

## Cosmology and P(k) ([`cosmo_utils.py`](../src/foraois/cosmo_utils.py) -- `CosmoData`)

Hubble parameter and matter density:

```
H(z)      = H0 * sqrt(OmegaM*(1+z)^3 + OmegaK*(1+z)^2 + OmegaLambda)
Omega_m(z) = OmegaM*(1+z)^3 / (H(z)/H0)^2
```

Linear growth factor `D(z)`, growth rate `f(z)`, collapse threshold `delta_col(z)` (normalized so `D(0)=1`):

```
a = 1/(1+z)
integral(a) = int_{a_min}^{a} da'/(a'*H(a'))^3
x = a*H(a)
D_raw(z) = (5/2)*Omega_m(z)*x^3*integral(a)
D(z)     = D_raw(z) / D_raw(z=0)
f(z)     = -1 - Omega_m(z)/2 + (1-Omega_m(z)) + 5*Omega_m(z)/(2*D_raw(z)/a)
delta_col(z) = 1.686 * D(z_pk) / D(z)      (delta_col(z_pk) = 1.686)
```
`z_pk` is the redshift at which `P(k)` and `sigma(M)` are evaluated: `CosmoData(params, redshift=[z_pk])` (only the first entry is used; default `[0.0]`). The barrier is normalised to the same epoch, so `delta_col(z)/sigma(M)`, the only combination a tree depends on, does not depend on the choice of `z_pk`. For `z_pk = 0` this is the usual `delta_col(z) = 1.686/D(z)`. 

Tabulated once at init on a uniform z-grid (`z_max=15`, `nz=5000` default) and looked up by interpolation; `pch_trees.py` guards that trees never grow past this table's `z_max`.

Cosmic time (relative units, `1/H0`):

```
t(a) = int_{a_min}^{a} da'/(a'*H(a'))
```

**Linear power spectrum** `P(k,z)`: computed via CLASS (`Run.mode: class`) or CAMB (`Run.mode: class`/`camb`, optional extras -- see README), on a `pk_kmin`..`pk_kmax` log-k grid of `pk_npoints` points. A dark-matter transfer function is then applied regardless of backend:

```
P(k) <- P(k) * T(k)^2
```

**sigma(M) and dlnsigma/dlnM**:

- top-hat / gaussian windows: Romberg-integrated,
  ```
  sigma^2(R) = (1/2*pi^2) * int k^3 P(k) W(kR)^2 dk
  dlnsigma/dlnM = (1/3) * R * (dsigma/dR) / sigma^2
  ```
- sharp-k window: closed-form/quadrature (chosen over Romberg because the window's discontinuity otherwise introduces ~3% integration error),
  ```
  sigma^2(R) = (1/2*pi^2) * int_0^{k0} k^2 P(k) dk,   k0 = alpha/R
  dlnsigma/dlnM = -k0^3 P(k0) / (6*pi^2*sigma^2)      (clamped to 0 once k0 > pk_kmax)
  ```

Mass-radius relation: `R(M) = (3M / 4*pi*rho_bar)^(1/3)`, `rho_bar = OmegaM * rhocrit0`, `rhocrit0 = 2.775e11 Msun/Mpc^3/h^2`.

### Numerical validity of sigma(M) and alpha(M) at low mass

`sigma(M)` and `alpha(M) = |dln(sigma)/dln(M)|` are looked up two ways: `sigma_at_logmass`/`dlogsigma_at_logmass` (scipy interpolants, extrapolate outside their tabulated range) and, inside the compiled (numba) tree kernels, a direct clamping lookup on the raw `_logmass`/`_sigma`/`_dlogsigma_dlogmass` arrays built by `_prepare_sigma_grid` (default range `10^2`-`10^16` Msun/h). The table starts at `10^2` Msun/h, and every public tree-building method calls `CosmoData.check_M_res(M_res, ...)` first, which **raises** `ValueError` if a build would still look sigma up below the table (rather than silently clamping or extrapolating arbitrarily far below a range that was never validated).

`sigma(M)`/`alpha(M)` are integrals of `P(k)` truncated at `pk_kmax`, and are systematically too small below a mass set by `pk_kmax`. Calibrated for a Planck 2018 CDM cosmology (top-hat window) against a `k_max=3e4` reference:

| `pk_kmax` [h/Mpc] | M for which sigma, alpha are accurate to 1% | to 3% | to 5% |
| ---: | ---: | ---: | ---: |
| 100  | `>= 8e7` | `>= 2.0e7` | `>= 1.6e7` |
| 300  | `>= 8.9e5` | `>= 7.1e5` | `>= 5.6e5` |
| 1000 | `>= 2.2e4` | `>= 1.8e4` | `>= 1.4e4` |
| 3000 | `>= 8e2`   | `>= 6.3e2` | `>= 5.6e2` |

`alpha`, not `sigma`, sets these limits (by a factor of ~8-13); both scale as `M_min ~ (pk_kmax/1000)^-3.0`, consistent with the geometric expectation `pk_kmax * R(M) ~ const`, with `pk_kmax * R(M_min) ~ 4.0` (1%), `3.7` (3%), `3.4` (5%). `check_M_res` **warns** (not raises -- this degrades gracefully rather than returning nonsense) when `pk_kmax * R(M_res) < 4.1`, i.e. when 1% pointwise accuracy has not been established -- **this is not a claim that the calculation is invalid**: many tree-level statistics do not need 1% sigma/alpha accuracy at every mass, and the warning message states the pk_kmax needed for 3%/5% too, so the 1%-boundary trigger is a conservative default rather than the only threshold that matters. It is a practical criterion in the same spirit as the `N_upper ~ 0.1` timestep criterion below, not an exact validity boundary; it is skipped for thermal-relic WDM and standard (top-hat) FDM, whose transfer functions suppress `P(k)` at the relevant `k` (making the CDM threshold spuriously conservative there), and for the sharp-k window, whose `alpha` is already forced to exactly 0 below `M(k0=pk_kmax)` by the closed form above.

**Tree-level statistics can need more margin than this pointwise sigma/alpha criterion alone.** A `z0=5`-anchored history integrated to `z_max=25` (`menon_power_2024.yml`) needed `pk_kmax=3000`, not the ~1500 the table above would suggest, to pass a full battery of tree statistics (median main-progenitor mass, formation redshift, resolved fraction, merger count, each at 1% or better) at `M_res=10^4` Msun/h; a `z0=0` grid to `z_max=20` at the same `M_res` was already adequate at `pk_kmax=1000`. The dependence on the specific `(z0, z_max, dz)` configuration was not characterised beyond these two cases. `menon_power_2024.yml` ships with `pk_kmax=3000` for this reason; the other shipped configs (`pk_kmax=100`, used at `M_res >= 10^9` throughout the README/tests/quick-start) are unaffected and unchanged. See the audit (`foraois_sigma_audit/REPORT.md`, `REPORT_kmax.md` in the maintainer's records) for the full tree-level tables and the derivation above.

## Transfer functions ([`transfer_functions.py`](../src/foraois/transfer_functions.py))

Selected by `Run.dm_model`, applied inside `get_power_spectrum()` as `P(k) *= T(k)^2`.

| Model | Config | Formula |
|---|---|---|
| CDM (default) | `dm_model: cdm` (or omitted) | `T(k) = 1` |
| WDM (thermal relic, Viel et al. 2005) | `dm_model: wdm`, `dm_model_mass`: mass in keV | `T(k) = [1 + (alpha*k)^(2*nu)]^(-5/nu)`, `nu=1.12`, `alpha = 0.049*(m_wdm/1keV)^-1.11*(Omega_wdm/0.25)^0.11*(h/0.7)^1.22` [h^-1 Mpc]. Calibrated for k < 5 h/Mpc. |
| FDM (fuzzy/scalar-field, Hu, Barkana & Gruzinov 2000) | `dm_model: fdm`, `dm_model_mass`: mass in units of 1e-22 eV | `T(k) = cos(x^3)/(1+x^8)`, `x = 1.61*m_a22^(1/18)*k_phys/k_Jeq`, `k_Jeq = 9*m_a22^0.5` [physical Mpc^-1], `k_phys = k*h`. Linear power-spectrum suppression only -- does **not** modify the collapse barrier (see [FDM caveat](#fdm-caveat) below). |

Half-mode wavenumbers (`wdm_half_mode_k`, `fdm_half_mode_k`) solve `T(k_hm) = 1/sqrt(2)` for each model (closed-form for WDM; `brentq` root-find on the monotonic branch for FDM).

## Collapse barrier ([`collapse/barrier.py`](../src/foraois/collapse/barrier.py) -- `delta_c(M, z, cosmo_data)`)

The collapse barrier is a separate choice from the dark-matter model. The dark-matter model (`dm_model`) modifies `P(k)` and hence `sigma(M)` inside `CosmoData`; the barrier is selected by the config keys `barrier` and `barrier_parameter` and read from `cosmo_data.run_params`. Tree-building code calls `delta_c(M, z, cosmo_data)` and does not build the barrier itself. (The earlier form `delta_c(M, z, model, cosmo_data)` is still accepted; its `model` argument is ignored.) The `model=` argument of the tree functions is likewise retained for backwards compatibility only and does not select the barrier.

| `barrier` | `delta_c(M, z)` | Notes |
|---|---|---|
| `fixed` (default) | `delta_sc(z)` = `CosmoData.delta_col_at_z(z)` | Mass-independent spherical-collapse threshold. `barrier_parameter` is ignored. |
| `linear` | `delta_sc(z) + beta * sigma^2(M)`, `beta = barrier_parameter` (default 0.15, must be >= 0) | Linear in the variance `S = sigma^2(M)`, so `B(S) = a + beta*S`, whose first-crossing density is known analytically (`first_crossing.linear_barrier_first_crossing`). It rises towards small masses. **Phenomenological and illustrative**: a controlled test of the general first-crossing path, not a physically calibrated collapse model. `beta = 0.15` is a chosen default, not a prediction. `linear` with `beta = 0` is the fixed barrier. |

For `linear`, masses below the floor of the `sigma(M)` table (100 Msun/h by default) are evaluated at the floor. This matters because the first-crossing grid (`S_max = S_max_factor * S_res`) can extend to variances the table does not represent (with a finite `pk_kmax`, `sigma^2` saturates at small mass), where the inverse mass map underflows; without the floor the barrier became `inf` and `first_crossing_step` returned NaN. Beyond the table's largest variance the barrier is therefore constant, so `B(S)` is not exactly `a + beta*S` there; this affects only the far tail of `f(S)` (part of the unresolved fraction `F_zh`), and was not quantified.

Unknown `barrier` names raise `ValueError`. `config/planck2018_camb.yml` sets `barrier: fixed`; `config/planck2018_linear_barrier.yml` is the same configuration with `barrier: linear`.

**Which algorithms support which barrier.** A scale-dependent barrier (`linear` with `beta > 0`) is supported only by the general serial Zhang-Hui path. Every other algorithm assumes a flat barrier and raises `NotImplementedError` (naming the requested barrier) rather than silently using the fixed one:

| Barrier | PCH08 | ZH `build_tree` (serial) | ZH `build_forest_numpy` | ZH `build_forest_numba` / `grow_full_population_numba` | Constrained trees | `expected_eps_splits_per_step` |
|---|---|---|---|---|---|---|
| `fixed`, or `linear` with `beta = 0` | works | works | works | works | works | works |
| `linear`, `beta > 0` | error | works | error | error | error | error |

PCH08 reads `delta_col(z)` directly, the NumPy/Numba forest builders use the closed-form flat-barrier sampler, and the constrained construction assumes a flat effective barrier, so none of them can represent a mass-dependent barrier. There is no general vectorised or compiled moving-barrier sampler; the serial path is the reference implementation.

The dark-matter model's own caveats belong to `CosmoData`: `dm_model='fdm'` warns that only the linear power-spectrum suppression is modelled (see the [FDM caveat](#fdm-caveat)), and `dm_model='sidm'` raises `NotImplementedError`.

## Merger tree generation ([`pch_trees.py`](../src/foraois/pch_trees.py) -- `PCHMergerTree`)

Implements the Parkinson, Cole & Helly (2008) Appendix A branching-rate/rejection-sampling algorithm exactly, restricted to their `gamma1 >= 0` branch (best-fit constants `G0=0.57`, `gamma1=0.38`, `gamma2=-0.01`, hard-coded).

**Timestep compliance.** PCH08 requires the expected number of resolved splits per step (`N_upper`) to be small, because a step registers at most one split. We adopt `N_upper` (PCH08) or `E` (Zhang-Hui) of about 0.1 or below as a practical criterion for this single-split construction, not a mathematical validity boundary. `N_upper` rises as `M_res / M0` falls, so a fixed `dz` that is adequate at `M_res / M0 = 1e-2` can be far too coarse at `1e-4` (max `N_upper` = 14 at `dz = 0.05`, 0.14 at `dz = 0.0005` for `M0 = 1e12`, `M_res / M0 = 1e-4`). When `N_upper >> 1` the main-progenitor histories become close to deterministic and assemble too early. Check `foraois.diagnostics.expected_splits_per_step` and reduce `dz` until the statistic of interest is stable. The Zhang-Hui builder is also limited to one split per step; compare its step with the EPS expected number of splits per step (`foraois.diagnostics.expected_eps_splits_per_step`). See [PCH08_HIGH_Z_DIAGNOSTIC.md](PCH08_HIGH_Z_DIAGNOSTIC.md) for the table. The paper's conditional-mass-function and major-merger figures use `M_res / M2 = 1e-4` through the adaptive full-tree grower in `scripts/paper_figs/_treegrowth.py`, which adapts the step to `N_upper`.

At each step, for a halo of mass `M2` at `sigma2 = sigma(M2)`, with `M_res` the mass resolution:

```
qres = M_res/M2
V(q) = sigma(qM)^2 / (sigma(qM)^2 - sigma2^2)^1.5
beta = ln(V(0.5)/V(qres)) / ln(0.5/qres)
B    = V(0.5) / 0.5^beta
mu   = alpha_h = |dlnsigma/dlnM|(M2/2)
eta  = beta - 1 - gamma1*mu
S_coeff = sqrt(2/pi) * B * alpha_h * G0 * 2^(-mu*gamma1) * (delta_col(z)/sigma2)^gamma2 * (sigma(M2/2)/sigma2)^gamma1
Nupper  = S_coeff * dz(dt/dz factor) * (0.5^eta - qres^eta)/eta     (0 if M2 < 2*M_res)
```

Progenitor mass ratio `q` is drawn from the envelope distribution above (closed-form inverse-CDF), then accepted/rejected against the exact PCH08 rate via

```
R(q) = (alpha(qM)/alpha_h) * (V(q)/(B*q^beta)) * ((2q)^mu * sigma(qM)/sigma(M2/2))^gamma1
```

The unresolved-mass fraction lost below `M_res` each step (eq. A6/A7):

```
u_res = sigma2 / sqrt(sigma(M_res)^2 - sigma2^2)
J(u_res) = int_0^{u_res} (1 + 1/u^2)^(gamma1/2) du      (tabulated once)
F = sqrt(2/pi) * J(u_res) * (G0/sigma2) * (delta_col(z)/sigma2)^gamma2 * dz-factor,  clipped to [0,1]
```

Per step: draw `r1`; no split if `r1 > Nupper`, mass becomes `M0*(1-F)`. Otherwise draw a candidate `q`, compute `R(q)`, draw `r3`; reject (no split) if `r3 > R(q)`; otherwise split into `q*M0` and `M0*(1-F-q)`, keeping the more massive as the main progenitor.

**Three backends** (`--backend`): `serial` (single-tree loop, reference/debug), `numpy` (all trees stepped through each redshift bin at once, no extra deps, default), `numba` (JIT + `nb.prange`-parallel across CPU cores -- fastest for large tree counts, but does not return per-step split events; requires the optional `numba` dependency, `pip install foraois[numba]`).

`build_tree`/`build_forest_*` track only the main branch; `build_full_tree` grows the entire branching structure (every progenitor, capped at `max_nodes`) for full merger-tree visualization (e.g. dendrograms).

| Parameter | Where | Meaning |
|---|---|---|
| `M0` | CLI / caller | Root halo mass at `z0` (Msun/h) |
| `z0`, `z_max` | CLI / caller | Start and end redshift of tree growth |
| `M_res` | CLI / caller | Mass resolution below which progenitors are not resolved (Msun/h) |
| `dz` | CLI / caller (default 0.1) | Fixed redshift step, shared across all halos in a forest -- a simplification vs. PCH08's per-halo adaptive step; use `diagnostics.py` to check `Nupper` stays small |
| `--n_trees` | CLI | Forest size (number of independent trees) |
| `--backend` | CLI | `serial` / `numpy` / `numba` |

## First-crossing distributions ([`first_crossing.py`](../src/foraois/first_crossing.py))

Implements the Zhang & Hui (2006) first-crossing solution (a numerical solution of their Volterra equation; closed form for a constant barrier) for a Markovian excursion-set walk (valid for a **sharp-k window** only) with an arbitrary moving barrier `B(S)` -- not wired into `PCHMergerTree` (its Appendix A rate has no defined meaning for a non-fitted barrier), but is the basis of `zhang_hui_trees.py`, below.

```
P0(delta,S) = exp(-delta^2/2S) / sqrt(2*pi*S)
f(S) = g1(S) + int_0^S f(S') g2(S,S') dS'
g1(S) = (B(S)/S - 2*dB/dS) * P0(B(S), S)
g2(S,S') = [2*dB/dS - (B(S)-B(S'))/(S-S')] * P0(B(S)-B(S'), S-S')
```

`solve_first_crossing(B, S_max, N, dBdS=None)` discretizes to `N+1` points and solves by forward substitution (O(N^2)), with a dedicated near-diagonal treatment for the singular `g2` term. Validated against closed-form flat/linear-barrier solutions and an independent Monte Carlo random walk.

## Zhang-Hui tree generation ([`zhang_hui_trees.py`](../src/foraois/zhang_hui_trees.py) -- `ZhangHuiMergerTree`)

The unfitted counterpart to `PCHMergerTree`: instead of PCH08's fitted `(G0, gamma1, gamma2)` rate, samples one binary split per step from `first_crossing.py`'s first-crossing distribution, evaluated against the configured barrier `collapse.delta_c(M, z, cosmo_data)`. The binary-per-step construction is an approximation to the EPS tree, and like PCH08 it needs a step small enough that the EPS expected number of splits per step is small (`foraois.diagnostics.expected_eps_splits_per_step`). This section summarizes the current API and its known limitations.

Origin-shift convention (Bond, Cole, Efstathiou & Kaiser 1991): a progenitor of a halo `(M0, z0)` at target redshift `z1` is a first crossing of the shifted barrier `B(S) = delta_c(M(S), z1) - delta_c(M0, z0)` at `S = sigma(M(S))^2 - sigma(M0)^2`. For the fixed barrier `delta_c` is mass-independent, so `B(S)` reduces to the constant `delta_c(z1) - delta_c(z0)` -- the closed-form Press-Schechter/inverse-Gaussian case. For the linear barrier `B(S) = delta_c(z1) - delta_c(z0) + beta*S`.

Three backends, differing in generality vs. speed:

- **`build_tree`** (serial, single halo): general -- works for any barrier `solve_first_crossing` can handle, and is the only ZH path that supports the scale-dependent `linear` barrier. O(`N_grid`^2) per step; `first_crossing_step` warns if `N_grid` is too coarse relative to the barrier's own scale `B(0)^2` (not simply `S_res`, a `sigma(M)`-derived scale that can be much larger). See "Resolution of the general first-crossing path" below for how the accuracy depends on that ratio.
- **`build_forest_numpy`** (vectorized): only for a flat barrier (`fixed`, or `linear` with `beta = 0`), checked explicitly from the config (`require_flat_barrier`) and then numerically (`_assert_flat_barrier`). Exploits that a flat barrier's first-crossing distribution is a **Levy distribution** with a closed-form CDF/inverse (`erfc`/`erfcinv`), so no numerical solve is needed at all -- `~0.06s` for 50,000 trees over 5 steps. Raises `NotImplementedError` for a scale-dependent barrier.
- **`build_forest_numba`** (JIT + `nb.prange`-parallel): same flat-barrier restriction and closed-form Levy sampling as `build_forest_numpy`, but stepped per-halo inside a `@njit(parallel=True)` kernel rather than array-vectorized -- `scipy.special.erfcinv` isn't callable from jitted code, so the kernel carries its own numba-compatible inverse-`erfc` (Winitzki approximation + Newton-Raphson refinement against `math.erf`, accurate to ~1e-6 relative error outside the extreme tails). Roughly 7x faster than `build_forest_numpy` for large forests. Needs `pip install foraois[numba]`. **Not seed-reproducible**: numba's `parallel=True`/`nb.prange` gives each worker thread its own internal RNG stream that `np.random.seed()` does not control, whether seeded outside the jitted function or as its own first statement -- this is a genuine numba limitation (also present, previously undocumented, in `PCHMergerTree.build_forest_numba`), not a bug specific to this kernel. Use `build_forest_numpy` when bitwise reproducibility across runs matters.

### Resolution of the general first-crossing path

For a scale-dependent barrier the serial path solves the first-crossing equation numerically (`solve_first_crossing`), and its accuracy depends on resolution. For the linear barrier the answer is known analytically, which gives a direct check. The comparison below is for `first_crossing_step`, the path `build_tree` uses.

**Finding.** The discrepancy is discretisation error that decreases with resolution; no implementation error was identified.
- `first_crossing_step` gives the same error as `solve_first_crossing` supplied with the exact barrier and derivative on the same grid, so the mass mapping, the numerical derivative and the step machinery add nothing.
- With `beta = 0` the solution agrees with the analytic flat-barrier result to machine precision.
- The error grows roughly in proportion to `beta` at fixed resolution (measured for `beta = 0.15` and `0.4` on the test fixture), and it decreases at about order 1/2 in `N_grid` in the tested regime (roughly a factor 0.71 per doubling). The reason for that order was not investigated.
- The accuracy is set by the grid spacing `dS = S_max/N_grid` relative to the barrier scale `a^2 = B(0)^2 = (delta_sc(z1) - delta_sc(z0))^2`, which sets the width of the peak of `f(S)`.

**Tested regime and measured accuracy.** `beta = 0.15`, `M0 = 1e12`, `M_res = 1e11` Msun/h, `z0 = 0`, `S_max = 2 S_res`. Errors are the peak-normalised error of `f(S)` over `0.05 < S <= S_res`, and the absolute error of the crossing probability at `S_res` (the `p_res` that `build_tree` uses). The maximum pointwise relative error is not used, because it is dominated by the tail and can change sign as `N_grid` grows.

Real CAMB spectrum (`planck2018_linear_barrier.yml`; reproduce with `scripts/validate_linear_barrier_convergence.py`):

| z1 | a^2 | N_grid | dS/a^2 | peak-normalised error | crossing-probability error |
|---|---|---|---|---|---|
| 2.0 | 5.51 | 100 | 0.015 | 6.1e-3 | 1.1e-3 |
| 2.0 | 5.51 | 400 | 0.004 | 3.1e-3 | 5.4e-4 |
| 2.0 | 5.51 | 1600 | 0.001 | 1.5e-3 | 2.7e-4 |
| 0.5 | 0.254 | 100 | 0.33 | 7.7e-3 | 1.0e-2 |
| 0.5 | 0.254 | 400 | 0.082 | 3.0e-3 | 2.8e-3 |
| 0.5 | 0.254 | 1600 | 0.021 | 1.4e-3 | 1.3e-3 |

At `z1 = 2` both errors fall monotonically. At `z1 = 0.5` the peak-normalised error falls monotonically, while the crossing-probability error is not monotone for `N_grid <~ 400` (it dips near `N_grid ~ 200` before rising, consistent with the signed error changing sign; the signed error was not inspected) and falls monotonically beyond that.

**Coarse-grid regime.** When `dS/a^2` is of order 1 or larger the result is badly wrong, and `f(S)` and `p_res` can be non-physical (`f` negative, `p_res` slightly negative); `first_crossing_step` warns when `dS > a^2`. For example, at `z1 = 0.5` with `M_res = 1e10` and default `N_grid = 400`, `S_max_factor = 8` the pointwise error reached about 82%. On the synthetic-spectrum test fixture at `z1 = 0.5`, the crossing-probability error is 0.18 at `dS/a^2 = 0.86`, 0.04 at 0.43, and 2e-3 at 0.054 (`tests/test_zhang_hui_validation.py`).

**Empirical guide, not a requirement.** In the tested regime, a few times 1e-3 in the quantities above needs roughly `dS/a^2 <~ 0.1`, i.e. `N_grid >~ 10 S_max/a^2`. This is a measured result for these setups, not a general theorem, and the code does not enforce it. The existing `first_crossing_step` warning fires only when `dS > a^2`, which is about ten times coarser than this. Because `a^2 = (delta_sc(z1) - delta_sc(z0))^2` shrinks with the step, smaller `dz` needs a finer grid at fixed accuracy, and the cost per step grows as `N_grid^2`. For `M0 = 1e12`, `M_res = 1e11` and `S_max = 2 S_res`, the guide gives `N_grid` of about 15 at `dz = 2`, 330 at `dz = 0.5`, 2,400 at `dz = 0.2` and 41,000 at `dz = 0.05`. No `dz` restriction is imposed.

**Resolved limitation (was open, closed via a literature re-read):** a drawn progenitor mass `M2` originally had no guaranteed lower bound on its complement (`M0*(1-F_zh) - M2`, the "continuing" branch) the way PCH08's `q`-range restriction guarantees both split fragments stay resolved -- this complement could land below `M_res`, or even go negative, in **over half of steps** in a direct check at `dz=0.2`. Checked directly against N23's own footnote 3: their progenitor mass `M'` is drawn from `[M_res, M-M_res]` -- bounded away from *both* ends -- which by construction keeps both `M'` and `M-M'` `>= M_res`. Both backends now draw from the matching restricted range (`first_crossing_step`'s `S_lower`/`p_split`) rather than the unrestricted `[0, S_res]` -- checked directly, `0` of `221` two-progenitor draws landed below `M_res` afterward. `smooth_accretion` is still computed as the conservation residual (`parent_mass - max(progenitors) - merger_mass` in `build_tree`; the array equivalent in `build_forest_numpy`), kept as a cheap consistency check, but now provably equals the simple `F_zh*M` formula rather than folding in a real leftover.

Comparing CDM output against `PCHMergerTree` at matched `(M0, z0, z_max, M_res, dz)` on a real Planck cosmology (`scripts/validate_zhang_hui_vs_pch08.py`, `n_trees=20000`, `M0=1e12`, `M_res=1e10`, `dz=0.005`): both survival fractions are 1.000 and the mean surviving mass of Zhang-Hui is about 19% below PCH08's, stable at 18-19% for `dz` from 0.01 to 0.002. The origin of the residual difference is not established. N23 do not expect agreement here (their unconstrained trees are not recalibrated against PCH08 or N-body). Re-run the script for the current value.

## Halo mass functions ([`mass_function_utils.py`](../src/foraois/mass_function_utils.py))

```
n(M,z) = (rho_bar/M^2) * f(nu) * |dlnnu/dlnM|,   nu = delta_col(z)/sigma(M),   rho_bar = Omega_m(z)*2.7755e11
```

| Model | `f(nu)` |
|---|---|
| Press-Schechter | `sqrt(2/pi) * nu * exp(-nu^2/2)` |
| Sheth-Tormen | `0.322*(1+nu'^-0.6)*f_PS(nu')`, `nu' = 0.84*nu` |

## Window functions ([`utils/window_function.py`](../src/foraois/utils/window_function.py))

| Window | `W(kR)` | Notes |
|---|---|---|
| top-hat | `3*j1(kR)/(kR)` | Exact spherical Bessel or a stable series approximation (`use_spherical_bessel`) |
| gaussian | `exp(-0.5*(kR)^2)` | |
| sharp-k | `1` if `kR <= alpha`, else `0` | Avoids the top-hat's "cloud-in-cloud" leakage above a WDM/FDM suppression scale; derivative w.r.t. R is a Dirac delta, so `sigma(M)`'s mass-derivative uses the closed form in `cosmo_utils.py` instead |

| Parameter | Config key | Default | Meaning |
|---|---|---|---|
| window_function_type | `Run.window_function_type` | `top_hat` | `top_hat` / `gaussian` / `sharp_k` |
| sharp_k_alpha | `Run.sharp_k_alpha` | 2.5 | Sharp-k cutoff `k0 = alpha/R` (Benson et al. 2013 / Kulkarni & Ostriker 2022 calibration) |
| use_spherical_bessel | `Run.use_spherical_bessel` | False | Exact Bessel vs. series approximation for the top-hat window |

Note: `M_res` should stay well above whatever suppression scale the chosen `dm_model` + window combination imposes -- `sigma(M)` genuinely flattens below it, and the PCH08 branching-rate algebra is poorly conditioned there.

## Config reference ([`utils/io.py`](../src/foraois/utils/io.py), [`config/`](../config))

A run is configured by a YAML file with top-level `Run`, `Cosmology`, and a section matching `Run.mode` (`class`/`camb`/`user`).

**`Cosmology`**

| Key | Meaning | Units |
|---|---|---|
| `H0` | Hubble constant | km/s/Mpc |
| `OmegaBar` | Baryon density Omega_b | -- |
| `OmegaM` | Total matter density Omega_m | -- |
| `OmegaK` | Curvature (optional, default 0.0) | -- |
| `As` | Primordial scalar amplitude | -- |
| `ns` | Scalar spectral index | -- |
| `tau_reio` | Reionization optical depth | -- |
| `mnu` | Total neutrino mass (optional, default 0.0) | eV |
| `num_massive_neutrinos` | (optional, default 0.0) | -- |

**`Run`**

| Key | Default | Meaning |
|---|---|---|
| `mode` | -- (required) | `class` / `camb` / `user` (Boltzmann backend; `user` supplies your own tabulated P(k), no extra dependency) |
| `dm_model` | `cdm` | `cdm` / `wdm` / `fdm` |
| `dm_model_mass` | `None` | Required for `wdm` (keV) / `fdm` (1e-22 eV units) |
| `barrier` | `fixed` | Collapse barrier, independent of `dm_model`: `fixed` (`delta_sc(z)`) or `linear` (`delta_sc(z) + beta*sigma^2(M)`, illustrative). See "Collapse barrier" above. |
| `barrier_parameter` | `0.15` | `beta` for `linear` (must be >= 0); ignored for `fixed` |
| `pk_kmin`, `pk_kmax` | 1e-4, 1e2 | Power spectrum k-range, h/Mpc -- `pk_kmax` sets how far down in mass `sigma(M)`/`alpha(M)` are accurate; see "Numerical validity of sigma(M) and alpha(M) at low mass" above before using `M_res` below ~1e7 Msun/h |
| `pk_npoints` | 1000 | Number of log-k grid points |
| `plot_pk`, `plot_mvar` | False | Generate P(k) / sigma(M) diagnostic plots |
| `pk_file_name`, `mvar_file_name` | -- | Output plot filenames |
| `use_spherical_bessel` | False | See window functions above |
| `window_function_type` | `top_hat` | See window functions above |
| `sharp_k_alpha` | 2.5 | See window functions above |

**`class`**: `output` (e.g. `mPk`), `P_k_max_1/Mpc` (max k for CLASS, physical 1/Mpc). **`camb`**: no extra keys needed -- CAMB cosmology params are derived from the shared `Cosmology` block. **`user`**: `pk_file` (required) -- path to a two-column, whitespace/comma-separated text table (`k` in h/Mpc, `P(k)` in `(Mpc/h)^3`, no header; `#`-prefixed comment lines are fine), log-log linearly interpolated onto the `pk_kmin`..`pk_kmax` grid (`CosmoData._load_user_power_spectrum`). Requesting k outside the table's own range raises rather than extrapolating -- widen the table or narrow `pk_kmin`/`pk_kmax`. `As`/`ns`/`tau_reio` are still required fields in `Cosmology` (the shared derivation logic reads them unconditionally) but are unused for this mode, since P(k) is supplied directly rather than generated from primordial parameters -- any placeholder value works. No CLASS/CAMB dependency needed at all, so this is the one path that uses only foraois's core (non-optional) dependencies end to end.

**Example configs** in [`config/`](../config): `planck2018.yml` (CDM, CLASS backend), `planck2018_camb.yml` (CDM, CAMB backend), `planck2018_user.yml` (same CDM cosmology, `mode: user`, reading `example_user_pk.txt` -- itself generated via CAMB, illustrative rather than a substitute for running your own Boltzmann code), `planck2018_wdm.yml` (3 keV thermal-relic WDM, CAMB), `planck2018_fdm.yml` (1e-22 eV FDM, CAMB), `planck2018_fdm_sharpk.yml` (same FDM mass with the sharp-k window, `sharp_k_alpha: 2.5`).

## CLI ([`main.py`](../src/foraois/main.py))

```
python -m foraois.main --params_file <config.yml> --n_trees 1000 --backend numpy
```

| Flag | Default | Meaning |
|---|---|---|
| `--params_file` | required | Path to the YAML config |
| `--n_trees` | 1000 | Forest size |
| `--algorithm` | `pch08` | `pch08` / `zhang-hui` / `constrained` |
| `--backend` | `numpy` | `serial` / `numpy` / `numba` (needs `pip install foraois[numba]`); ignored for `--algorithm constrained` |
| `--M1`, `--z1` | 1e11, 4.0 | `--algorithm constrained` only: the guaranteed progenitor mass/redshift |

`--algorithm zhang-hui`/`constrained` both call `ZhangHuiMergerTree.build_tree`'s `O(N_grid^2)`-per-step solver somewhere in their path (directly for `zhang-hui --backend serial`; via the unconstrained continuation for `constrained`) -- `main.py` uses a coarser `(z_max=8, dz=0.2, N_grid=60)` there than PCH08's `(z_max=15, dz=0.005)`, and neither is meant for bulk (`--n_trees` large) generation the way PCH08's `numpy`/`numba` backends or `zhang-hui --backend numpy` (closed-form, no such cost) are.

## FDM caveat

Current FDM support (`transfer_functions.T_FDM`) captures only the *linear power-spectrum suppression* of fuzzy dark matter. Some literature (e.g. Du et al. 2017) argues a mass-dependent collapse barrier is also needed to reproduce simulated FDM halo mass functions at the low-mass end; the current code applies the configured collapse barrier (the constant CDM barrier `delta_col(z) = 1.686/D(z)` unless `barrier: linear` is requested) regardless of `dm_model`; no FDM-specific barrier is implemented. Deriving a Schrodinger-Poisson-motivated, mass-dependent `delta_c(M,z)` is open research, see [ROADMAP.md](../ROADMAP.md).

## References

See [README.md](../README.md#references) for the full citation list.

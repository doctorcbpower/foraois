# foraois: models and parameter reference

This document lists the exact equations implemented in [`src/foraois/`](../src/foraois) and every config parameter that controls them. For module responsibilities and usage, see [README.md](../README.md). Wavenumbers `k` are in h/Mpc unless stated otherwise; masses in Msun/h; `h = H0/100`.

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
delta_col(z) = 1.686 / D(z)      (delta_col(z=0) = 1.686 hard-coded)
```
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

## Transfer functions ([`transfer_functions.py`](../src/foraois/transfer_functions.py))

Selected by `Run.dm_model`, applied inside `get_power_spectrum()` as `P(k) *= T(k)^2`.

| Model | Config | Formula |
|---|---|---|
| CDM (default) | `dm_model: cdm` (or omitted) | `T(k) = 1` |
| WDM (thermal relic, Viel et al. 2005) | `dm_model: wdm`, `dm_model_mass`: mass in keV | `T(k) = [1 + (alpha*k)^(2*nu)]^(-5/nu)`, `nu=1.12`, `alpha = 0.049*(m_wdm/1keV)^-1.11*(Omega_wdm/0.25)^0.11*(h/0.7)^1.22` [h^-1 Mpc]. Calibrated for k < 5 h/Mpc. |
| FDM (fuzzy/scalar-field, Hu, Barkana & Gruzinov 2000) | `dm_model: fdm`, `dm_model_mass`: mass in units of 1e-22 eV | `T(k) = cos(x^3)/(1+x^8)`, `x = 1.61*m_a22^(1/18)*k_phys/k_Jeq`, `k_Jeq = 9*m_a22^0.5` [physical Mpc^-1], `k_phys = k*h`. Linear power-spectrum suppression only -- does **not** modify the collapse barrier (see [FDM caveat](#fdm-caveat) below). |

Half-mode wavenumbers (`wdm_half_mode_k`, `fdm_half_mode_k`) solve `T(k_hm) = 1/sqrt(2)` for each model (closed-form for WDM; `brentq` root-find on the monotonic branch for FDM).

## Collapse barrier ([`collapse/`](../src/foraois/collapse) -- `delta_c(M, z, model, cosmo_data)`)

The single interface for the spherical-collapse threshold, dispatching to a per-model implementation. Not consumed by `pch_trees.py` (see the module's own docstring): PCH08's Appendix A algorithm is derived for a barrier that depends only on `z`, and CDM/WDM (the only models with real content below) are mass-independent anyway, so there is nothing for that tree-building code to gain from routing through this. It is used by `zhang_hui_trees.py`, whose exact-rate algorithm is barrier-agnostic.

| Model | Function | Formula | Status |
|---|---|---|---|
| CDM | `delta_c_cdm` | `delta_col(z)` (existing `CosmoData.delta_col_at_z`) | Done; `M` accepted, unused (constant-in-M by construction) |
| WDM | `delta_c_wdm` | Same as CDM | Same constant barrier as CDM -- whether this is sufficient without a WDM-specific PCH08 rate refit (Benson et al. 2013's approach) is an open question, see [ROADMAP.md](../ROADMAP.md) |
| FDM | `delta_c_fdm` | Falls back to the CDM value | **Known-inadequate placeholder** -- warns on every call. The real mass-dependent moving barrier is still open research, see [ROADMAP.md](../ROADMAP.md) |
| SIDM | `delta_c_sidm` | -- | Always raises `NotImplementedError` -- no barrier scoped yet |

## Merger tree generation ([`pch_trees.py`](../src/foraois/pch_trees.py) -- `PCHMergerTree`)

Implements the Parkinson, Cole & Helly (2008) Appendix A branching-rate/rejection-sampling algorithm exactly, restricted to their `gamma1 >= 0` branch (best-fit constants `G0=0.57`, `gamma1=0.38`, `gamma2=-0.01`, hard-coded).

At each step, for a halo of mass `M2` at `sigma2 = sigma(M2)`, with `M_res` the mass resolution:

```
qres = M_res/M2
V(q) = sigma(qM)^2 / (sigma(qM)^2 - sigma2^2)^1.5
beta = ln(V(0.5)/V(qres)) / ln(0.5/qres)
B    = V(0.5) / 0.5^beta
mu   = alpha_h = |dlnsigma/dlnM|(M2/2)
eta  = beta - 1 - gamma1*mu
S_coeff = sqrt(2/pi) * B * alpha_h * G0 * 2^(mu*gamma1) * (delta_col(z)/sigma2)^gamma2 * (sigma(M2/2)/sigma2)^gamma1
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

Implements the Zhang & Hui (2006) exact first-crossing solution for a Markovian excursion-set walk (valid for a **sharp-k window** only) with an arbitrary moving barrier `B(S)` -- not wired into `PCHMergerTree` (its Appendix A rate has no defined meaning for a non-fitted barrier), but is the basis of `zhang_hui_trees.py`, below.

```
P0(delta,S) = exp(-delta^2/2S) / sqrt(2*pi*S)
f(S) = g1(S) + int_0^S f(S') g2(S,S') dS'
g1(S) = (B(S)/S - 2*dB/dS) * P0(B(S), S)
g2(S,S') = [2*dB/dS - (B(S)-B(S'))/(S-S')] * P0(B(S)-B(S'), S-S')
```

`solve_first_crossing(B, S_max, N, dBdS=None)` discretizes to `N+1` points and solves by forward substitution (O(N^2)), with a dedicated near-diagonal treatment for the singular `g2` term. Validated against closed-form flat/linear-barrier solutions and an independent Monte Carlo random walk.

## Zhang-Hui exact-rate tree generation ([`zhang_hui_trees.py`](../src/foraois/zhang_hui_trees.py) -- `ZhangHuiMergerTree`)

The barrier-agnostic counterpart to `PCHMergerTree`: instead of PCH08's fitted `(G0, gamma1, gamma2)` rate, samples progenitor masses directly from `first_crossing.py`'s exact solution, evaluated against `collapse.delta_c(M, z, model, cosmo_data)`. This section summarizes the current API and its known limitations.

Origin-shift convention (Bond, Cole, Efstathiou & Kaiser 1991): a progenitor of a halo `(M0, z0)` at target redshift `z1` is a first crossing of the shifted barrier `B(S) = delta_c(M(S), z1) - delta_c(M0, z0)` at `S = sigma(M(S))^2 - sigma(M0)^2`. For CDM/WDM (and FDM's current placeholder), `delta_c` is mass-independent, so `B(S)` reduces to the constant `delta_c(z1) - delta_c(z0)` -- the closed-form Press-Schechter/inverse-Gaussian case.

Two backends, differing in generality vs. speed:

- **`build_tree`** (serial, single halo): general -- works for any barrier `solve_first_crossing` can handle, including a future genuinely mass-dependent one. O(`N_grid`^2) per step; `first_crossing_step` warns if `N_grid` is too coarse relative to the barrier's own scale `B(0)^2` (not simply `S_res`, a `sigma(M)`-derived scale that can be much larger).
- **`build_forest_numpy`** (vectorized): only for models whose barrier is checked (at runtime, via `_assert_flat_barrier`) to be flat -- currently `cdm`/`wdm`/`fdm`. Exploits that a flat barrier's first-crossing distribution is a **Levy distribution** with a closed-form CDF/inverse (`erfc`/`erfcinv`), so no numerical solve is needed at all -- `~0.06s` for 50,000 trees over 5 steps. Raises `NotImplementedError` for `sidm` or any future non-flat barrier.

**Resolved limitation (was open, closed via a literature re-read):** a drawn progenitor mass `M2` originally had no guaranteed lower bound on its complement (`M0*(1-F_zh) - M2`, the "continuing" branch) the way PCH08's `q`-range restriction guarantees both split fragments stay resolved -- this complement could land below `M_res`, or even go negative, in **over half of steps** in a direct check at `dz=0.2`. Checked directly against N23's own footnote 3: their progenitor mass `M'` is drawn from `[M_res, M-M_res]` -- bounded away from *both* ends -- which by construction keeps both `M'` and `M-M'` `>= M_res`. Both backends now draw from the matching restricted range (`first_crossing_step`'s `S_lower`/`p_split`) rather than the unrestricted `[0, S_res]` -- checked directly, `0` of `221` two-progenitor draws landed below `M_res` afterward. `smooth_accretion` is still computed as the conservation residual (`parent_mass - max(progenitors) - merger_mass` in `build_tree`; the array equivalent in `build_forest_numpy`), kept as a cheap consistency check, but now provably equals the simple `F_zh*M` formula rather than folding in a real leftover.

Comparing CDM output against `PCHMergerTree` at matched `(M0, z0, z_max, M_res, dz)` on a real Planck cosmology (`scripts/validate_zhang_hui_vs_pch08.py`, `n_trees=20000`): survival fraction identical; mean surviving mass currently runs **~11.4% lower** than PCH08's (an initial ~21% dropped to ~4.4% once the grid-resolution bug was fixed, then to ~11.4% once the mass-budget/sampling-range bug above was *properly* fixed rather than just patched with a residual -- the fix removed a real upward bias, not noise) -- N23 explicitly do not expect exact agreement here (their unconstrained trees are not recalibrated against PCH08/N-body), so this is reported, not treated as a defect to eliminate.

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
| `mode` | -- (required) | `class` / `camb` / `user` (Boltzmann backend; `user` not yet implemented) |
| `dm_model` | `cdm` | `cdm` / `wdm` / `fdm` |
| `dm_model_mass` | `None` | Required for `wdm` (keV) / `fdm` (1e-22 eV units) |
| `pk_kmin`, `pk_kmax` | 1e-4, 1e2 | Power spectrum k-range, h/Mpc |
| `pk_npoints` | 1000 | Number of log-k grid points |
| `plot_pk`, `plot_mvar` | False | Generate P(k) / sigma(M) diagnostic plots |
| `pk_file_name`, `mvar_file_name` | -- | Output plot filenames |
| `use_spherical_bessel` | False | See window functions above |
| `window_function_type` | `top_hat` | See window functions above |
| `sharp_k_alpha` | 2.5 | See window functions above |

**`class`**: `output` (e.g. `mPk`), `P_k_max_1/Mpc` (max k for CLASS, physical 1/Mpc). **`camb`**: no extra keys needed -- CAMB cosmology params are derived from the shared `Cosmology` block.

**Example configs** in [`config/`](../config): `planck2018.yml` (CDM, CLASS backend), `planck2018_camb.yml` (CDM, CAMB backend), `planck2018_wdm.yml` (3 keV thermal-relic WDM, CAMB), `planck2018_fdm.yml` (1e-22 eV FDM, CAMB), `planck2018_fdm_sharpk.yml` (same FDM mass with the sharp-k window, `sharp_k_alpha: 2.5`).

## CLI ([`main.py`](../src/foraois/main.py))

```
python -m foraois.main --params_file <config.yml> --n_trees 1000 --backend numpy
```

| Flag | Default | Meaning |
|---|---|---|
| `--params_file` | required | Path to the YAML config |
| `--n_trees` | 1000 | Forest size |
| `--backend` | `numpy` | `serial` / `numpy` / `numba` (needs `pip install foraois[numba]`) |

## FDM caveat

Current FDM support (`transfer_functions.T_FDM`) captures only the *linear power-spectrum suppression* of fuzzy dark matter. Some literature (e.g. Du et al. 2017) argues a mass-dependent collapse barrier is also needed to reproduce simulated FDM halo mass functions at the low-mass end; the current code still applies the constant CDM barrier `delta_col(z) = 1.686/D(z)` regardless of `dm_model`. Deriving a Schrodinger-Poisson-motivated, mass-dependent `delta_c(M,z)` is open research, see [ROADMAP.md](../ROADMAP.md).

## References

See [README.md](../README.md#references) for the full citation list.

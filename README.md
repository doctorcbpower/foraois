## Foraois
[![CI](https://github.com/doctorcbpower/foraois/actions/workflows/ci.yml/badge.svg)](https://github.com/doctorcbpower/foraois/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Monte Carlo dark matter halo merger trees, for a range of dark matter models

Foraois (Irish, "forest") generates Monte Carlo merger trees for dark matter
halo growth histories, with two interchangeable algorithms:

- **`PCHMergerTree`** -- Parkinson, Cole & Helly (2008)'s fitted branching-rate
  approach, with three interchangeable performance backends (`serial`/`numpy`/`numba`).
- **`ZhangHuiMergerTree`** -- builds trees directly from the *exact* Zhang &
  Hui (2006) first-crossing distribution rather than a fitted rate, so it's
  barrier-agnostic by construction. Includes a working implementation of
  Nadler, Benson, Driskell, Du & Gluscevic (2023)'s Brownian-bridge-constrained
  excursions: merger trees guaranteed to reach a specified progenitor mass at
  a specified redshift, useful for cheaply sampling rare/outlier growth
  histories without down-sampling a huge unconstrained ensemble.

Both support CDM, WDM, and FDM cosmologies via power-spectrum transfer
functions and window-function choice. See [docs/MODELS.md](docs/MODELS.md)
for the governing equations behind every module, and [ROADMAP.md](ROADMAP.md)
for what's not yet covered.

Cosmology-specific physics (the linear power spectrum, the mass variance
sigma(M), and the linear collapse threshold delta_col(z)) is computed once in
`CosmoData` and handed to the tree-growth kernels as precomputed lookup
tables. Because the fast kernels only ever consume those tables -- never
`P(k)` directly -- any dark matter model that can be expressed as a modified
power spectrum/window function (e.g. WDM, fuzzy/scalar-field DM) plugs in at
the `CosmoData` layer without touching the tree-building code itself.

## Installation

```
git clone git@github.com:doctorcbpower/foraois.git
cd foraois
pip install -e .
```

Core dependencies (`numpy`, `scipy`, `pyyaml`, `numba`, `matplotlib`) install automatically. A linear power spectrum is computed via CLASS or CAMB, which are optional extras -- install whichever backend(s) you need:

```
pip install -e .[class]   # requires classy (compiled from source)
pip install -e .[camb]    # pip-installable
```

(A `user` mode for supplying your own tabulated `P(k)` without either dependency is planned but not yet implemented -- see `CosmoData.get_power_spectrum()`.)

The example configs under `config/` (used throughout this README and the demo notebook) are part of the git checkout, not shipped with the package itself -- the commands below assume you're running from the repository root. Not yet published to PyPI.

## Quick start

**PCH08, via the CLI** -- the fastest way to generate a large forest of trees:

```
python -m foraois.main --params_file config/planck2018.yml --n_trees 1000 --backend numba
```

**PCH08, via Python** -- for programmatic use, the same algorithm:

```python
from foraois.cosmo_utils import CosmoData
from foraois.pch_trees import PCHMergerTree
from foraois.utils import io

run_params = io.get_params("config/planck2018.yml")
cosmo_data = CosmoData(run_params, redshift=[0.0])
tree_generator = PCHMergerTree(cosmo_data, run_params)

tree = tree_generator.build_tree(M0=1e12, z0=0.0, z_max=5.0, M_res=1e9, dz=0.2)
```

**The exact Zhang-Hui algorithm** -- same tree-of-dicts shape as `build_tree` above, any collapse barrier:

```python
from foraois.zhang_hui_trees import ZhangHuiMergerTree

tree_generator = ZhangHuiMergerTree(cosmo_data, run_params, model="cdm")
tree = tree_generator.build_tree(M0=1e12, z0=0.0, z_max=5.0, M_res=1e9, dz=0.2)
```

**A constrained tree**, guaranteed to reach a specified `(M1, z1)`:

```python
from foraois.zhang_hui_constrained_trees import build_constrained_tree

tree = build_constrained_tree(
    M0=1e12, z0=0.0, M1=1e11, z1=4.0, z_max=8.0, M_res=1e9,
    cosmo_data=cosmo_data, model="cdm",
)
```

See [notebooks/foraois_demo.ipynb](notebooks/foraois_demo.ipynb) for a full interactive walkthrough of all three, and [docs/MODELS.md](docs/MODELS.md) for the API/equations behind each.

## Usage

Configure cosmological parameters and code options in a YAML file (see `config/planck2018.yml` for an example: a top-level `Run` section selects `mode: class|camb|user` and power-spectrum grid settings, `Cosmology` sets the cosmological parameters, and `class`/`camb` hold backend-specific options), then run:

To use a non-standard dark matter model, add `dm_model`/`dm_model_mass` to the `Run` section -- this multiplies the linear `P(k)` by a transfer function before anything else touches it, so no other configuration changes are needed. Omit `dm_model` (or set it to `cdm`) for standard CDM, the default.

* `dm_model: wdm`, `dm_model_mass: <mass in keV>` (see `config/planck2018_wdm.yml`) -- thermal-relic WDM (Viel et al. 2005).
* `dm_model: fdm`, `dm_model_mass: <mass in 1e-22 eV>` (see `config/planck2018_fdm.yml`) -- fuzzy/scalar-field DM (Hu, Barkana & Gruzinov 2000). This captures FDM's *linear* power-spectrum suppression only -- not the mass-dependent collapse barrier some literature (e.g. Du et al. 2017) argues is also needed to reproduce simulated FDM halo mass functions at the low-mass end; see `transfer_functions.py`'s module docstring.

The window function used for `sigma(M)` is independent of `dm_model` and defaults to the real-space top-hat. Add `window_function_type: sharp_k` (and optionally `sharp_k_alpha`, default `2.5`) to the `Run` section to use a sharp-k (Fourier-space step-function) window instead -- `W(k,R) = 1` for `k <= alpha/R`, `0` otherwise (Benson et al. 2013's WDM calibration, also used for FDM by Kulkarni & Ostriker 2022; see `config/planck2018_fdm_sharpk.yml`). This avoids the top-hat's "cloud-in-cloud" artifact, where its oscillatory tails let some small-scale power leak back in above a WDM/FDM suppression scale that a top-hat window doesn't fully remove. Because the window is discontinuous, `CosmoData` computes both `sigma(M)` and its mass derivative for `sharp_k` via dedicated closed-form/quadrature methods rather than the generic numerical-derivative route used for `top_hat`/`gaussian` -- transparent to any caller, but worth knowing if you're reading `cosmo_utils.py`. Note the resolution mass `M_res` passed to tree-building should stay well above whatever suppression scale a given `dm_model`+window combination imposes: `sigma(M)` genuinely flattens out below it (there's essentially no power left to resolve), and PCH08's branching-rate algebra isn't well-conditioned in that flat regime regardless of window choice.

```
python -m foraois.main --params_file config/planck2018.yml --n_trees 1000 --backend numba
```

`--backend` selects the tree-growth kernel:

| Backend | Description |
|---|---|
| `serial` | Original single-tree loop (`PCHMergerTree.build_tree`) -- kept for debugging/reference |
| `numpy` | Vectorised NumPy, all trees stepped through each redshift bin simultaneously, no extra dependencies |
| `numba` | JIT + `nb.prange` parallel across all CPU cores (default, fastest for large `N`) |

## Package structure

| Module | Responsibility |
|---|---|
| `cosmo_utils.py` | `CosmoData`: linear growth factor, `delta_col(z)`, cosmic time, power spectrum (CLASS/CAMB), mass variance `sigma(M)`; applies `transfer_functions` for non-CDM models |
| `transfer_functions.py` | `T_CDM`, `T_WDM(k, m_wdm, ...)` (Viel et al. 2005 thermal-relic fit) with `wdm_half_mode_k`, and `T_FDM(k, m_a22, h)` (Hu, Barkana & Gruzinov 2000) with `fdm_half_mode_k`; applied as `T(k)^2` to `P(k)` inside `CosmoData.get_power_spectrum()` |
| `pch_trees.py` | `PCHMergerTree`: the three vectorised/main-progenitor-only tree-building backends (implementing Parkinson, Cole & Helly 2008 Appendix A's branching-rate/rejection-sampling algorithm exactly), plus `build_full_tree` for growing a single tree's complete branching structure (every progenitor, not just the main branch) |
| `diagnostics.py` | Validation tools for the branching-rate machinery: step-size safety (`expected_splits_per_step`) and sampling correctness (`check_sampling_consistency`) against an independent quadrature reference |
| `first_crossing.py` | `solve_first_crossing`: exact first-crossing distribution f(S) for a Markovian random walk with an arbitrary barrier B(S) (Zhang & Hui 2006), solved via forward substitution -- no Monte Carlo needed. Not wired into `PCHMergerTree` (its Appendix A rate has no defined meaning for a non-fitted barrier); is the basis of `zhang_hui_trees.py`, below |
| `collapse/` | `delta_c(M, z, model, cosmo_data)`: collapse-barrier interface, dispatching to per-model implementations (`cdm`, `wdm` -- constant-in-M; `fdm` -- warned placeholder falling back to CDM; `sidm` -- not implemented, raises). Not wired into `pch_trees.py` (PCH08's own fitted rate doesn't use it), but is what `zhang_hui_trees.py`/`zhang_hui_constrained_trees.py` build on (see docs/MODELS.md's "Collapse barrier" section) |
| `zhang_hui_trees.py` | `draw_progenitor_mass_zh`/`ZhangHuiMergerTree.build_tree` -- serial, general (any barrier) tree-building on `first_crossing.py`'s exact Zhang & Hui rate, mirroring `PCHMergerTree.build_tree`'s signature/return shape; `ZhangHuiMergerTree.build_forest_numpy` -- vectorised, closed-form (Levy-distribution) forest builder for the currently-flat-barrier models (`cdm`/`wdm`/`fdm`'s placeholder), the barrier-agnostic counterpart to PCH08's fitted rate |
| `first_crossing_constrained.py` | `solve_first_crossing_constrained`: N23's Brownian-bridge-constrained first-crossing distribution (their Eq. A5), via direct midpoint discretization (Du et al. 2017), a different numerical method from `first_crossing.py`'s own; `simulate_bridge_path`: barrier-free sequential Brownian-bridge sampling |
| `zhang_hui_constrained_trees.py` | `constrained_branch_growth_history`: a single branch guaranteed to reach `(M1, z1)`, via `simulate_bridge_path` + successive-maxima extraction + a `delta_c`-inversion for redshift -- barrier-agnostic by construction, `smooth_accretion`/`merger_mass` tracked per step; `build_constrained_tree`: grafts this onto `ZhangHuiMergerTree.build_tree`'s unconstrained continuation beyond `z1`, same tree-of-dicts schema throughout. See [ROADMAP.md](ROADMAP.md) for what's not yet covered here (a vectorised forest backend, secondary progenitor sub-trees on the constrained portion) |
| `tree_algorithm.py` | `TreeAlgorithm`: a `typing.Protocol` formalizing the `build_tree`/`build_forest_numpy` interface `PCHMergerTree` and `ZhangHuiMergerTree` both satisfy |
| `mass_function_utils.py` | Press-Schechter / Sheth-Tormen halo mass functions |
| `utils/window_function.py` | `WindowFunctions`: top-hat, Gaussian, and sharp-k (`k0=alpha/R`, Benson et al. 2013) window functions for `sigma(M)` |
| `utils/io.py` | YAML config loading |
| `utils/plot.py` | Publication-style plots (white background, colourblind-safe palette): mass tracks, merger rate, mass function, main-progenitor tree graph, `plot_dendrogram` (a full branching-tree diagram in the style of Nadler et al. 2023 Figure 4, built from `build_full_tree`), and `plot_excursion_trajectory` (a halo growth history re-expressed as an excursion-set random walk, Nadler et al. 2023 Figure 1 style) |
| `main.py` | CLI entry point |

## Interactive exploration

`notebooks/foraois_demo.ipynb` walks through cosmology setup, `PCHMergerTree`'s three tree-building backends, the built-in visualisations (including a Figure-4-style full merger-tree diagram via `build_full_tree`/`plot_dendrogram`, and a Figure-1-style excursion-set trajectory view via `plot_excursion_trajectory`), the `diagnostics` validation tools, CDM-vs-WDM and CDM-vs-FDM comparisons (`config/planck2018_wdm.yml`, `config/planck2018_fdm.yml`, `plot_dm_model_comparison`), and a top-hat-vs-sharp-k window comparison for FDM (`config/planck2018_fdm_sharpk.yml`) -- together showing that tree-building itself needs no changes at all for a non-CDM model or a non-default window function. Open it with `jupyter notebook notebooks/foraois_demo.ipynb` after `pip install -e .[camb] jupyter`.

**Not yet covered by the notebook, a real gap**: `ZhangHuiMergerTree` (exact rate) and the Brownian-bridge-constrained backend -- both are Python-API-only and demonstrated so far only in the test suite and validation scripts (`scripts/validate_zhang_hui_vs_pch08.py`, `scripts/validate_constrained_tree_convergence.py`), not in an interactive/tutorial form.

## Testing

```
pip install pytest
pytest tests/
```

152 tests as of this writing, all using a synthetic power-law `P(k)` by default (no CLASS/CAMB installation required to run the suite), organized roughly as:

**PCH08 backend.** `test_pch_trees.py` covers all four `PCHMergerTree` tree-building methods (shape/bounds, mass monotonicity, NumPy-backend seed reproducibility, NumPy/Numba statistical consistency, `build_full_tree`'s branching structure); `test_pch_validation.py` is the branching-rate validation battery (the `J(u)` lookup table, step-size diagnostics, Monte-Carlo-vs-quadrature sampling-consistency checks).

**Cosmology/transfer functions.** `test_cosmo_utils.py` covers `CosmoData`'s math (growth factor, `delta_col`, cosmic time, `sigma(M)`, radius/mass conversions, `logmass_at_sigma`'s inverse) and the sharp-k window (step-function shape, the closed-form `d ln(sigma^2)/d ln M`, config wiring, end-to-end tree building). `test_transfer_functions.py`'s pure-math checks (`T_WDM`/`T_FDM` shape/limits/monotonicity) need no cosmology backend; its integration checks (real `P(k)`/`sigma(M)` suppression, end-to-end `dm_model='wdm'`/`'fdm'` tree building) use real CAMB via `pytest.importorskip`, so they're skipped rather than failing if CAMB isn't installed.

**Collapse barrier.** `test_collapse_barrier.py` covers `foraois.collapse.delta_c`'s dispatch to `cdm`/`wdm`/`fdm`/`sidm`, including `fdm`'s disclosed-placeholder warning and `sidm`'s `NotImplementedError`.

**Zhang & Hui exact rate (unconstrained).** `test_first_crossing.py` validates `solve_first_crossing` against two independent closed-form solutions (flat and linear barriers, to machine precision and ~0.5% respectively) and a direct Monte Carlo random walk for a genuinely nonlinear barrier (reproducing Zhang & Hui 2006's own Figure 2 cross-check) -- pure random-walk math, no cosmology backend needed. `test_zhang_hui_trees.py` covers `ZhangHuiMergerTree`'s `build_tree`/`build_forest_numpy` (structure, mass conservation -- including the regression tests for the sampling-range fix that guarantees both fragments of a resolved split stay above `M_res`, matching N23's own footnote 3). `test_zhang_hui_validation.py` cross-checks the actual barrier construction against a direct Monte Carlo random walk, independent of the solver's own internals.

**Brownian-bridge-constrained excursions (N23's own novel contribution).** `test_first_crossing_constrained.py` validates the constrained first-crossing solver against its own core numerical method (checked on the unconstrained case, which has a closed form), the `S1 -> infinity` convergence-to-unconstrained limit, and an independent rejection-sampling Monte Carlo cross-check. `test_zhang_hui_constrained_trees.py` covers the barrier-free bridge-path sampler (cross-checked against the solver above), `constrained_branch_growth_history`'s exact-endpoint guarantee and mass conservation, and `build_constrained_tree`'s grafting onto the unconstrained continuation.

**Backend interface.** `test_tree_algorithm.py` verifies `PCHMergerTree` and `ZhangHuiMergerTree` both actually satisfy the shared `TreeAlgorithm` interface, rather than just asserting it.

**Plotting.** `test_plot.py` is smoke tests for every `utils/plot.py` function.

## References

This code implements the algorithms in:

* Parkinson, Cole & Helly 2008, **Generating dark matter halo merger trees**, _MNRAS_, 383, 557. [DOI: 10.1111/j.1365-2966.2007.12517.x](https://doi.org/10.1111/j.1365-2966.2007.12517.x)
* Nadler, Benson, Driskell, Du & Gluscevic 2023, **Growing the first galaxies' merger trees**, _MNRAS_, 521, 3201. [arXiv:2212.08584](https://arxiv.org/abs/2212.08584)
* Viel, Lesgourgues, Haehnelt, Matarrese & Riotto 2005, **Constraining warm dark matter candidates including sterile neutrinos and light gravitinos with WMAP and the Lyman-alpha forest**, _Phys. Rev. D_, 71, 063534 (thermal-relic WDM transfer function, eq. 6-7). [arXiv:astro-ph/0501562](https://arxiv.org/abs/astro-ph/0501562)
* Hu, Barkana & Gruzinov 2000, **Fuzzy Cold Dark Matter: The Wave Properties of Ultralight Particles**, _Phys. Rev. Lett._, 85, 1158 (FDM transfer function, eq. 8-9). [arXiv:astro-ph/0003365](https://arxiv.org/abs/astro-ph/0003365)
* Benson, Farahi, Cole, Moustakas, Jenkins, Lovell, Kennedy, Helly & Frenk 2013, **Dark matter halo merger histories beyond cold dark matter: I. Methods and application to warm dark matter**, _MNRAS_, 428, 1774 (sharp-k window calibration, `alpha=2.5`). [DOI: 10.1093/mnras/sts159](https://doi.org/10.1093/mnras/sts159)
* Kulkarni & Ostriker 2022, **What is the halo mass function in a fuzzy dark matter cosmology?**, _MNRAS_, 510, 1425 (sharp-k window applied to FDM, `alpha=2.5`, cross-checked against Benson et al. 2013 and Lacey & Cole 1994). [arXiv:2011.02116](https://arxiv.org/abs/2011.02116)
* Zhang & Hui 2006, **On Random Walks with a General Moving Barrier**, (exact first-crossing distribution via a Volterra integral equation, eq. 5, solved by forward substitution). [arXiv:astro-ph/0508384](https://arxiv.org/abs/astro-ph/0508384)

# foraois: package structure

| Module | Responsibility |
|---|---|
| `cosmo_utils.py` | `CosmoData`: linear growth factor, `delta_col(z)`, cosmic time, power spectrum (CLASS/CAMB/tabulated), mass variance `sigma(M)`; applies `transfer_functions` for non-CDM models |
| `transfer_functions.py` | `T_CDM`, `T_WDM(k, m_wdm, ...)` (Viel et al. 2005 thermal-relic fit) with `wdm_half_mode_k`, and `T_FDM(k, m_a22, h)` (Hu, Barkana & Gruzinov 2000) with `fdm_half_mode_k`; applied as `T(k)^2` to `P(k)` inside `CosmoData.get_power_spectrum()` |
| `pch_trees.py` | `PCHMergerTree`: the three vectorised/main-progenitor-only tree-building backends (implementing Parkinson, Cole & Helly 2008 Appendix A's branching-rate/rejection-sampling algorithm exactly), plus `build_full_tree` for growing a single tree's complete branching structure (every progenitor, not just the main branch) |
| `diagnostics.py` | Validation tools for the branching-rate machinery: step-size safety (`expected_splits_per_step`) and sampling correctness (`check_sampling_consistency`) against an independent quadrature reference |
| `first_crossing.py` | `solve_first_crossing`: first-crossing distribution f(S) (numerical solution of the Volterra equation; closed form for a constant barrier) for a Markovian random walk with an arbitrary barrier B(S) (Zhang & Hui 2006), solved via forward substitution -- no Monte Carlo needed. Not wired into `PCHMergerTree` (its Appendix A rate has no defined meaning for a non-fitted barrier); is the basis of `zhang_hui_trees.py`, below |
| `collapse/` | `delta_c(M, z, cosmo_data)`: the collapse-barrier prescription, selected by the config keys `barrier` (`fixed` = `delta_sc(z)`, or the illustrative `linear` = `delta_sc(z) + beta*sigma^2(M)`) and `barrier_parameter`, independent of `dm_model`; plus `barrier_is_flat`/`require_flat_barrier`, the explicit guards used by every algorithm that assumes a flat barrier (PCH08, the closed-form ZH samplers, the constrained sampler, `expected_eps_splits_per_step`). `zhang_hui_trees.py`/`zhang_hui_constrained_trees.py` build on `delta_c` (see [MODELS.md](MODELS.md)'s "Collapse barrier" section). The dark-matter model's own caveats (FDM placeholder warning, SIDM not implemented) live in `CosmoData._dm_transfer_function` |
| `zhang_hui_trees.py` | `draw_progenitor_mass_zh`/`ZhangHuiMergerTree.build_tree` -- serial, general (any barrier) tree-building on `first_crossing.py`'s Zhang & Hui first-crossing distribution (binary-per-step approximation), mirroring `PCHMergerTree.build_tree`'s signature/return shape; `ZhangHuiMergerTree.build_forest_numpy`/`build_forest_numba` -- vectorised NumPy and JIT-parallel Numba forest builders, both closed-form (Levy-distribution) for the currently-flat-barrier models (`cdm`/`wdm`/`fdm`'s placeholder), the unfitted counterpart to PCH08's fitted rate |
| `first_crossing_constrained.py` | `solve_first_crossing_constrained`: N23's Brownian-bridge-constrained first-crossing distribution (their Eq. A5), via direct midpoint discretization (Du et al. 2017), a different numerical method from `first_crossing.py`'s own; `simulate_bridge_path`: barrier-free sequential Brownian-bridge sampling |
| `zhang_hui_constrained_trees.py` | `constrained_branch_growth_history`: a single branch guaranteed to reach `(M1, z1)`, via `simulate_bridge_path` + successive-maxima extraction + a `delta_c`-inversion for redshift -- flat-barrier only (`require_flat_barrier`; the bridge construction assumes a flat effective barrier), `smooth_accretion`/`merger_mass` tracked per step; `build_constrained_tree`: grafts this onto `ZhangHuiMergerTree.build_tree`'s unconstrained continuation beyond `z1`, same tree-of-dicts schema throughout. See [ROADMAP.md](../ROADMAP.md) for what's not yet covered here (a vectorised forest backend, secondary progenitor sub-trees on the constrained portion) |
| `tree_algorithm.py` | `TreeAlgorithm`: a `typing.Protocol` formalizing the `build_tree`/`build_forest_numpy` interface `PCHMergerTree` and `ZhangHuiMergerTree` both satisfy |
| `mass_function_utils.py` | Press-Schechter / Sheth-Tormen halo mass functions |
| `utils/window_function.py` | `WindowFunctions`: top-hat, Gaussian, and sharp-k (`k0=alpha/R`, Benson et al. 2013) window functions for `sigma(M)` |
| `utils/io.py` | YAML config loading |
| `utils/plot.py` | Publication-style plots (white background, colourblind-safe palette): mass tracks, merger rate, mass function, main-progenitor tree graph, `plot_dendrogram` (a full branching-tree diagram in the style of Nadler et al. 2023 Figure 4, built from `build_full_tree`), and `plot_excursion_trajectory` (a halo growth history re-expressed as an excursion-set random walk, Nadler et al. 2023 Figure 1 style) |
| `main.py` | CLI entry point |

## Pipeline map

```
YAML / CosmoData.from_params
   -> CosmoData: growth, delta_sc(z);  P(k) (+ transfer function);  S(M) (+ window)
   -> collapse.delta_c(M, z, cosmo_data)   the barrier
   -> first_crossing_step                  composes B(S) from S(M) and the barrier, calls
                                           solve_first_crossing(B(S), ...), then does the Zhang-Hui bookkeeping
   -> tree construction: PCH08 | Zhang-Hui | constrained Zhang-Hui
```

`solve_first_crossing` receives only a callable `B(S)`; it knows nothing about cosmology or the configured barrier. S(M) and the barrier meet in `first_crossing_step`.

| Stage | Module | Principal symbols |
|---|---|---|
| Config loading | `utils/io.py` | `get_params`, `params_from_dict` (also validates `barrier`) |
| Cosmology | `cosmo_utils.py` | `CosmoData`, `delta_col_at_z`, `get_linear_growth_and_collapse` |
| P(k), transfer function | `cosmo_utils.py`, `transfer_functions.py` | `CosmoData.get_power_spectrum`, `_dm_transfer_function`; `T_WDM`, `T_FDM` |
| Mass variance S(M) | `cosmo_utils.py`, `utils/window_function.py` | `get_mass_variance`, `dlogsigma_dlogmass`, `prepare_sigma_grid` (built on first use), `sigma_at_logmass`, `logmass_at_sigma`; `WindowFunctions` |
| Barrier | `collapse/barrier.py` | `delta_c`, `barrier_settings`, `barrier_is_flat`, `require_flat_barrier` |
| General first crossing | `first_crossing.py` | `solve_first_crossing`, `flat_barrier_first_crossing`, `linear_barrier_first_crossing` |
| ZH composition and sampling step | `zhang_hui_trees.py` | `first_crossing_step`, `draw_progenitor_mass_zh`; closed-form flat-barrier helpers `_flat_barrier_cdf`, `_flat_barrier_sample`, `draw_progenitor_mass_zh_flat` |
| PCH08 trees | `pch_trees.py` | `PCHMergerTree` (`draw_progenitor_masses`, `build_tree`, `build_forest_*`, `build_full_tree`) |
| Zhang-Hui trees | `zhang_hui_trees.py` | `ZhangHuiMergerTree` (`build_tree`, `build_forest_numpy`, `build_forest_numba`, `grow_full_population_numba`) |
| Constrained construction | `first_crossing_constrained.py`, `zhang_hui_constrained_trees.py` | `simulate_bridge_path`; `constrained_branch_growth_history`, `build_constrained_tree`. The tree path does not call `solve_first_crossing_constrained`, which is a reference solver |
| Diagnostics | `diagnostics.py` | `expected_splits_per_step`, `expected_splits_per_step_zh`, `expected_eps_splits_per_step`, `true_split_probability`, `check_sampling_consistency` (step-size and sampling-consistency checks; mass conservation is tested in `tests/test_pch_trees.py`, `test_zhang_hui_trees.py` and `test_zhang_hui_constrained_trees.py`, not implemented in `diagnostics.py`) |

### Where the timestep enters

Through the `dz` argument of `build_tree`/`build_forest_*` on both tree classes (and `dz` of the `grow_full_population_*` methods). Zhang-Hui uses the supplied fixed `dz` throughout. PCH08's `build_tree` and `build_forest_*` also use the supplied fixed `dz`; per-halo adaptive step selection (PCH08 sec. 2.1) exists only in the compiled `grow_full_population_numba_adaptive` and `major_merger_redshifts_numba_adaptive` (`_pick_adaptive_step_scalar`).

### Where unresolved mass is handled

The two backends use different constructions and are not interchangeable:

- **Zhang-Hui:** `F_zh` (unresolved fraction), `S_lower` and `p_split` (resolved-split probability), computed in `first_crossing_step`. The flat-barrier paths repeat the same resolved/unresolved split logic in `draw_progenitor_mass_zh_flat`, `_build_forest_flat_barrier_numpy` and the Numba flat kernel, and `scripts/paper_figs/_treegrowth.py` mirrors it. A change to this prescription must be made in each.
- **PCH08:** `_unresolved_accretion_fraction` in `pch_trees.py`.

### Tree and forest representations

These are not interchangeable, and "full" means different things in different places.

| Representation | Produced by | Contents |
|---|---|---|
| Main-branch history (list of dicts) | `PCHMergerTree.build_tree`, `ZhangHuiMergerTree.build_tree` | One main progenitor per step; ZH also carries `smooth_accretion`/`merger_mass` |
| Main-branch ensemble | `build_forest_numpy`, `build_forest_numba` (both classes) | Arrays over many halos, including the accretion channels |
| Full branching tree | `PCHMergerTree.build_full_tree` only | All-progenitor node list for one tree |
| Branch-mass population at checkpoints | `PCHMergerTree.grow_full_population_numba_adaptive`, `ZhangHuiMergerTree.grow_full_population_numba` | `{checkpoint_z: [branch masses]}` per tree; no tree structure |
| Constrained main branch | `build_constrained_tree` | Main branch pinned to `(M1, z1)`, unconstrained continuation beyond it |

See [MODELS.md](MODELS.md) for the governing equations and parameter reference behind each module, and [../ROADMAP.md](../ROADMAP.md) for what's not yet covered.

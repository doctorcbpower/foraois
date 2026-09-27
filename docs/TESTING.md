# foraois: testing

```
pip install pytest
pytest tests/
```

300 tests as of this writing, all using a synthetic power-law `P(k)` by default (no CLASS/CAMB installation required to run the suite), organized roughly as:

**PCH08 backend.** `test_pch_trees.py` covers all four `PCHMergerTree` tree-building methods (shape/bounds, mass monotonicity, NumPy-backend seed reproducibility, NumPy/Numba statistical consistency, `build_full_tree`'s branching structure, and a regression test disclosing that `build_forest_numba` is *not* seed-reproducible -- see [MODELS.md](MODELS.md)); `test_pch_validation.py` is the branching-rate validation battery (the `J(u)` lookup table, step-size diagnostics, Monte-Carlo-vs-quadrature sampling-consistency checks).

**Cosmology/transfer functions.** `test_cosmo_utils.py` covers `CosmoData`'s math (growth factor, `delta_col`, cosmic time, `sigma(M)`, radius/mass conversions, `logmass_at_sigma`'s inverse), tabulated (`mode: user`) power-spectrum loading (including a real-CAMB round-trip fidelity check), and the sharp-k window (step-function shape, the closed-form `d ln(sigma^2)/d ln M`, config wiring, end-to-end tree building). `test_transfer_functions.py`'s pure-math checks (`T_WDM`/`T_FDM` shape/limits/monotonicity) need no cosmology backend; its integration checks (real `P(k)`/`sigma(M)` suppression, end-to-end `dm_model='wdm'`/`'fdm'` tree building) use real CAMB via `pytest.importorskip`, so they're skipped rather than failing if CAMB isn't installed.

**Collapse barrier.** `test_collapse_barrier.py` covers `foraois.collapse.delta_c` for the `fixed` and `linear` barriers (formula, defaults, beta dependence, beta = 0 equals fixed, validation, the config controlling the barrier, and the flatness guard). `test_io.py` covers the `barrier`/`barrier_parameter` config keys and the shared and linear-barrier config files. The FDM placeholder warning and the SIDM `NotImplementedError` are tested with the dark-matter transfer function in `test_transfer_functions.py`. The linear barrier's numerical validation (the general first-crossing path against the analytic solution, its convergence with resolution, and the regime where it is coarse) is in `test_zhang_hui_validation.py`. Each algorithm that assumes a flat barrier (PCH08, the ZH NumPy/Numba builders, the constrained sampler, `expected_eps_splits_per_step`) has a test that it rejects `beta > 0` and accepts `beta = 0`.

**Zhang & Hui exact rate (unconstrained).** `test_first_crossing.py` validates `solve_first_crossing` against two independent closed-form solutions (flat and linear barriers, to machine precision and ~0.5% respectively) and a direct Monte Carlo random walk for a genuinely nonlinear barrier (reproducing Zhang & Hui 2006's own Figure 2 cross-check) -- pure random-walk math, no cosmology backend needed. `test_zhang_hui_trees.py` covers `ZhangHuiMergerTree`'s `build_tree`/`build_forest_numpy`/`build_forest_numba` (structure, mass conservation -- including the regression tests for the sampling-range fix that guarantees both fragments of a resolved split stay above `M_res`, matching N23's own footnote 3; NumPy/Numba statistical consistency; the same numba-seed-non-reproducibility regression as PCH08's; a numba-internal `erfcinv` approximation checked directly against SciPy across the full domain). `test_zhang_hui_validation.py` cross-checks the actual barrier construction against a direct Monte Carlo random walk, independent of the solver's own internals.

**Brownian-bridge-constrained excursions (N23's own novel contribution).** `test_first_crossing_constrained.py` validates the constrained first-crossing solver against its own core numerical method (checked on the unconstrained case, which has a closed form), the `S1 -> infinity` convergence-to-unconstrained limit, and an independent rejection-sampling Monte Carlo cross-check. `test_zhang_hui_constrained_trees.py` covers the barrier-free bridge-path sampler (cross-checked against the solver above), `constrained_branch_growth_history`'s exact-endpoint guarantee and mass conservation, and `build_constrained_tree`'s grafting onto the unconstrained continuation.

**Backend interface.** `test_tree_algorithm.py` verifies `PCHMergerTree` and `ZhangHuiMergerTree` both actually satisfy the shared `TreeAlgorithm` interface, rather than just asserting it.

**CLI.** `test_main.py` is a smoke-test suite for every `--algorithm`/`--backend` combination `main.py` supports, plus its `--M1`/`--z1`/`--params_file` validation paths.

**Plotting.** `test_plot.py` is smoke tests for every `utils/plot.py` function.

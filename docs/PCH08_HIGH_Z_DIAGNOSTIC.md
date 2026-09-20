# PCH08 main-progenitor histories at small M_res and high redshift: a diagnostic note

Status: open question. This note records what was observed and what has and has not been established. It does not
conclude whether the behaviour is an implementation issue or a regime limitation.

## Observation

With `PCHMergerTree.build_forest_numba` (and `build_forest_numpy`), the main-progenitor mass history at small `M_res`
is nearly deterministic and assembles much earlier than the Zhang & Hui generator, and than mean accretion histories
(for example Correa et al. 2015 for the z0 = 0 case). `scripts/diagnose_pch08_high_z.py` reproduces the numbers below
(60 trees, `dz = 0.05`, masses in Msun/h; ratios only).

| Setup | PCH08, M(z)/M0 median [16, 84] | Zhang & Hui | Resolved merger, PCH08 vs ZH (fraction of steps) |
|---|---|---|---|
| z0 = 0, M0 = 1e12, M_res = 1e4, z = 1 | 0.840 [0.840, 0.840] | 0.449 [0.245, 0.614] | 2.2% vs 44.7% |
| same, z = 2 | 0.688 [0.688, 0.688] | 0.231 [0.140, 0.390] | 1.9% vs 42.2% |
| z0 = 5, M0 = 3e10, M_res = 1e4, z = 10 | 0.398 [0.398, 0.398] | 0.037 [0.015, 0.067] | 3.1% vs 30.2% |
| z0 = 5, M0 = 3e13, M_res = 1e4, z = 10 | 0.294 [0.294, 0.294] | 0.007 [0.003, 0.018] | 1.3% vs 50.8% |
| z0 = 0, M0 = 1e12, z = 1, M_res = 1e8 | 0.802 [0.782, 0.806] | 0.419 [0.279, 0.673] | 42.6% vs 37.4% |
| z0 = 0, M0 = 1e12, z = 1, M_res = 1e10 | 0.542 [0.338, 0.686] | 0.436 [0.262, 0.582] | 15.9% vs 16.5% |

## What is established

* The near-zero width of the PCH08 distribution and the early assembly are reproducible, at the standard z0 = 0 anchor as well
  as at z0 = 5.
* They do not depend on the backend (numpy and numba agree), on `dz` (0.02 to 0.2 changes M(z=10)/M0 by under 2%), or on the
  P(k)/barrier redshift convention (`CosmoData(redshift=[0.0])` and `[5.0]` give the same PCH08 numbers).
* At `M_res = 1e4` PCH08 resolves a merger in about 2 to 3% of steps and books almost all accreted mass as smooth accretion
  (merged/smooth mass ratio of order 1e-6 to 1e-4, against 3 to 6 for Zhang & Hui).
* PCH08 approaches Zhang & Hui only when `M_res` is around 1% of M0 (last row).

## What is not established

* Whether this is an implementation issue (for example in the treatment of fragments near or below `M_res`, the tabulated
  branching-rate grids, or the mass bookkeeping of the main branch) or an intrinsic limit of the PCH08 algorithm when `M_res / M0`
  is very small. PCH08's own calibration used much coarser resolution than 1e-8 of the halo mass.
* Whether `build_forest_numba` and `build_forest_numpy` return the same quantity as a `build_tree`-style main-progenitor history.
* Whether the earlier "PCH08 vs Zhang-Hui, about -11% in mean mass" validation used a resolution in the regime where the two agree
  (last row) rather than the small-`M_res` regime shown above.

## Suggested checks (none done)

1. Compare the accepted-split probability per step against the analytic PCH08 expectation as a function of `M_res / M0`.
2. Check the range of `logmass_grid` / `alpha_grid` / `j_u_grid` against fragment masses down to `M_res`.
3. Test the mass-conservation identity (`mass_history`, `smooth_accretion`, `merger_mass`) at very small `M_res`.
4. Decide whether to document a minimum `M_res / M0` for PCH08, or to add a guard that warns when it is below it.

## Downstream impact

Any use of PCH08 to follow assembly to high redshift with small `M_res` (as in Ashvini's high-z black-hole work, which needs
halos resolved down to about 1e4 Msun at z of order 20) should not assume it is equivalent to Zhang & Hui. Ashvini's own
comparison against PCH08 was therefore not possible and the old statement that it agreed to 0.02 to 0.14 dex cannot be reproduced.
